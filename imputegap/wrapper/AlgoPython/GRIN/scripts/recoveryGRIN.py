
"""
import copy
import datetime
import os
import pathlib

import numpy as np
import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR

from imputegap.wrapper.AlgoPython.GRIN.lib import fillers, datasets, config
from imputegap.wrapper.AlgoPython.GRIN.lib.data.imputation_dataset import GraphImputationDataset
from imputegap.wrapper.AlgoPython.GRIN.lib.nn import models
from imputegap.wrapper.AlgoPython.GRIN.lib.nn.utils.metric_base import MaskedMetric
from imputegap.wrapper.AlgoPython.GRIN.lib.nn.utils.metrics import MaskedMAE, MaskedMAPE, MaskedMSE, MaskedMRE
from imputegap.wrapper.AlgoPython.GRIN.lib.data.datamodule import SpatioTemporalDataModule


"""
def recoveryGRIN(input_data, seed=True, lr=0.001, epochs=300, patience=40, l2_reg=0.0,
                 grad_clip_val=5.0, grad_clip_algorithm="norm", loss_fn="l1_loss",
                 use_lr_schedule=True, adj_threshold=0.1, alpha=10.0, hint_rate=0.7,
                 g_train_freq=1, d_train_freq=5, val_len=0.1, test_len=0.2, window=12, stride=1):
    return "in progress"

"""
class NumPyDataset:

    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)

        # Automatically generate training and evaluation masks based on NaNs
        self.training_mask = ~np.isnan(self.data)
        self.eval_mask = np.ones_like(self.training_mask, dtype=bool)  # Assume all data can be evaluated

    @property
    def shape(self):
        return self.data.shape

    def numpy(self, return_idx=False):
        if return_idx:
            return self.data, np.arange(len(self.data))
        return self.data

    def splitter(self, torch_dataset, val_len=0.1, test_len=0.2):
        num_samples = len(torch_dataset)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        val_size = int(num_samples * val_len)
        test_size = int(num_samples * test_len)
        train_size = num_samples - val_size - test_size

        return indices[:train_size], indices[train_size:train_size + val_size], indices[train_size + val_size:]

    def get_similarity(self, thr=0.1):
        adj_matrix = np.random.rand(self.data.shape[1], self.data.shape[1])  # Random adjacency matrix
        return (adj_matrix > thr).astype(np.float32)  # Thresholded binary adjacency matrix


def recoveryGRIN(input_data, seed=True, lr=0.001, epochs=300, patience=40, l2_reg=0.0,
                 grad_clip_val=5.0, grad_clip_algorithm="norm", loss_fn="l1_loss",
                 use_lr_schedule=True, adj_threshold=0.1, alpha=10.0, hint_rate=0.7,
                 g_train_freq=1, d_train_freq=5, val_len=0.1, test_len=0.2, window=12, stride=1):
    # Set seed for reproducibility
    if seed:
        pl.seed_everything(42)

    torch.set_num_threads(1)

    model_cls, filler_cls = models.GRINet, fillers.GraphFiller

    ########################################
    # Prepare dataset                      #
    ########################################

    dataset = NumPyDataset(input_data)  # Converts NumPy array to a dataset that GRIN understands

    torch_dataset = GraphImputationDataset(
        *dataset.numpy(return_idx=True),  # Unpack the NumPy array properly
        mask=dataset.training_mask,
        eval_mask=dataset.eval_mask,
        window=window,
        stride=stride
    )

    train_idxs, val_idxs, test_idxs = dataset.splitter(torch_dataset, val_len=val_len, test_len=test_len)

    dm = SpatioTemporalDataModule(
        torch_dataset, train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=test_idxs
    )

    print("Scaling axis before setup:", dm.scaling_axis)  # Debugging line

    # Ensure a valid scaling axis is assigned
    print("Dataset shape:", dataset.shape)
    print("Dataset type:", type(dataset))
    print("First 5 rows of dataset:", dataset.data[:5])  # Print a sample of the data

    print("Valid scaling axes:", ["batch", "nodes", "time"])  # Debugging print
    print("Scaling axis before setup:", dm.scaling_axis)  # Debugging print

    # Try assigning "time" instead of "batch"
    dm.scaling_axis = "nodes"  # Change to "nodes" if "time" doesn't work

    dm.setup()

    adj = dataset.get_similarity(thr=adj_threshold)
    np.fill_diagonal(adj, 0.0)

    ########################################
    # Model Initialization                 #
    ########################################

    #additional_model_hparams = dict(adj=adj, d_in=dm.d_in)#, n_nodes=dm.n_nodes)

    additional_model_hparams = dict(
        adj=adj,
        d_in=dm.d_in,
        d_hidden=64,  # Set an appropriate hidden dimension (change if needed)
        d_ff=128,  # Set feedforward dimension (change if needed)
        ff_dropout=0.1  # Dropout value (change if needed)
    )

    loss_fn = MaskedMetric(metric_fn=getattr(F, loss_fn), metric_kwargs={'reduction': 'none'})
    metrics = {'mae': MaskedMAE(),
               'mape': MaskedMAPE(),
               'mse': MaskedMSE(),
               'mre': MaskedMRE()}

    scheduler_class = CosineAnnealingLR if use_lr_schedule else None

    filler = filler_cls(
        model_class=model_cls,
        model_kwargs=additional_model_hparams,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': lr, 'weight_decay': l2_reg},
        loss_fn=loss_fn,
        metrics=metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs={'eta_min': 0.0001, 'T_max': epochs},
        #alpha=alpha,
        #hint_rate=hint_rate,
        #g_train_freq=g_train_freq,
        #d_train_freq=d_train_freq
    )

    ########################################
    # Training                             #
    ########################################

    logdir = os.path.join(config['logs'], "custom_dataset", "GRIN", datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    pathlib.Path(logdir).mkdir(parents=True)

    early_stop_callback = EarlyStopping(monitor='val_mae', patience=patience, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1, monitor='val_mae', mode='min')

    logger = TensorBoardLogger(logdir, name="GRIN")

    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger,
        default_root_dir=logdir,
        gpus=1 if torch.cuda.is_available() else None,
        gradient_clip_val=grad_clip_val,
        gradient_clip_algorithm=grad_clip_algorithm,
        callbacks=[early_stop_callback, checkpoint_callback]
    )

    trainer.fit(filler, datamodule=dm)

    ########################################
    # Testing                              #
    ########################################

    filler.load_state_dict(torch.load(checkpoint_callback.best_model_path, lambda storage, loc: storage)['state_dict'])
    filler.freeze()
    trainer.test()
    filler.eval()

    if torch.cuda.is_available():
        filler.cuda()

    return "GRIN Recovery Completed"

"""