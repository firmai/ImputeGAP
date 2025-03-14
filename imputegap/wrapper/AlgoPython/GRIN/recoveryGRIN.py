import datetime
import os
import pathlib
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR

from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils
from imputegap.wrapper.AlgoPython.GRIN.lib.data.datamodule import SpatioTemporalDataModule
from imputegap.wrapper.AlgoPython.GRIN.lib.data.imputation_dataset import ImputationDataset, GraphImputationDataset
from imputegap.wrapper.AlgoPython.GRIN.lib.nn import models
from imputegap.wrapper.AlgoPython.GRIN.lib.nn.utils.metric_base import MaskedMetric
from imputegap.wrapper.AlgoPython.GRIN.lib.nn.utils.metrics import MaskedMAE, MaskedMAPE, MaskedMSE, MaskedMRE
from imputegap.wrapper.AlgoPython.GRIN.lib.utils import parser_utils

from imputegap.wrapper.AlgoPython.GRIN.lib import datasets, fillers, config


def has_graph_support(model_cls):
    return model_cls in [models.GRINet]


def get_model_classes(model_str):
    if model_str == 'grin':
        model, filler = models.GRINet, fillers.GraphFiller
    else:
        raise ValueError(f'Model {model_str} not available.')
    return model, filler


def recoveryGRIN(input, d_hidden=32, lr=0.001, batch_size=32, window=1, alpha=10.0, patience=4, epochs=20, workers=2,
                 adj_threshold=0.1, val_len=0.2, test_len=0.2, d_ff=16, ff_dropout=0.1, stride=1, l2_reg=0.0,
                 grad_clip_val=5.0, grad_clip_algorithm="norm", loss_fn="l1_loss", use_lr_schedule=True, hint_rate=0.7,
                 g_train_freq=1, d_train_freq=5, seed=42):

    print("\n\n\t\t\t(PYTHON) GRIN: Matrix Shape: (", input.shape[0], ", ", input.shape[1], ") for",
          " batch_size ", batch_size, " lr ", lr, " window ", window, " alpha ", alpha, " patience ", patience,
          " epochs ", epochs, ", and workers ", workers, "=================================================\n\n ")

    input_data = np.copy(input)

    if seed:
        seed = 42

    M, N = input_data.shape

    if window > N :
        window = N // 2

    torch.set_num_threads(seed)
    pl.seed_everything(seed)

    model_cls, filler_cls = get_model_classes('grin')
    dataset = datasets.MissingValuesMyData(input_data)

    ########################################
    # create logdir and save configuration #
    ########################################

    # Define split configuration
    split_conf = {
        "lr": lr,
        "epochs": epochs,
        "patience": patience,
        "l2_reg": l2_reg,
        "grad_clip_val": grad_clip_val,
        "grad_clip_algorithm": grad_clip_algorithm,
        "loss_fn": loss_fn,
        "use_lr_schedule": use_lr_schedule,
        "adj_threshold": adj_threshold,
        "alpha": alpha,
        "hint_rate": hint_rate,
        "g_train_freq": g_train_freq,
        "d_train_freq": d_train_freq,
        "val_len": val_len,
        "test_len": test_len,
        "window": window,
        "stride": stride,
        "d_hidden": d_hidden,  # Default or replace with correct value
        "d_ff": d_ff,  # Default or replace with correct value
        "ff_dropout": ff_dropout  # Default or replace with correct value
    }

    exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{seed}"
    logdir = os.path.join(config['logs'], 'grin', exp_name)
    # save config for logging
    pathlib.Path(logdir).mkdir(parents=True)

    ########################################
    # data module                          #
    ########################################

    # instantiate dataset
    dataset_cls = GraphImputationDataset if has_graph_support(model_cls) else ImputationDataset

    torch_dataset = dataset_cls(
        *dataset.numpy(return_idx=True),
        mask=dataset.training_mask,
        eval_mask=dataset.eval_mask,
        window=window,
        stride=stride,
    )

    # get train/val/test indices
    # ✅ MANUAL DATA SPLITTING (Replacing `splitter`)
    total_size = len(torch_dataset)
    indices = np.arange(total_size)

    # Shuffle indices before splitting
    np.random.shuffle(indices)

    test_size = int(total_size * test_len)
    val_size = int(total_size * val_len)
    train_size = total_size - val_size - test_size

    train_idxs = indices[:train_size]
    val_idxs = indices[train_size:train_size + val_size]
    test_idxs = indices

    # Check if indices are empty
    # Check if indices are empty
    print(f"🔍 torch size: {len(torch_dataset)}")
    print(f"🔍 Validation Indices: {len(val_idxs) if val_idxs is not None else 0}")
    print(f"🔍 Test Indices: {len(test_idxs) if test_idxs is not None else 0}")

    # Extract only the valid arguments that SpatioTemporalDataModule accepts
    data_module_conf = {
        "scale": True,
        "scaling_axis": "global",
        "scaling_type": "std",
        "scale_exogenous": None,
        "train_idxs": train_idxs,
        "val_idxs": val_idxs,
        "test_idxs": test_idxs,
        "batch_size": batch_size,
        "workers": workers,
        "samples_per_epoch": None
    }

    # Now, pass only the relevant parameters
    dm = SpatioTemporalDataModule(
        torch_dataset,
        **data_module_conf,
    )

    dm.setup()


    # get adjacency matrix
    adj = dataset.get_similarity(thr=adj_threshold)
    # force adj with no self loop
    np.fill_diagonal(adj, 0.0)

    ########################################
    # predictor                            #
    ########################################

    # model's inputs
    additional_model_hparams = dict(adj=adj, d_in=dm.d_in, n_nodes=dm.n_nodes)
    model_kwargs = parser_utils.filter_args(args={**split_conf, **additional_model_hparams},
        target_cls=model_cls  # ✅ Ensure target_cls is set correctly
    )

    # loss and metrics
    loss_fn = MaskedMetric(metric_fn=getattr(F, loss_fn), metric_kwargs={'reduction': 'none'})

    metrics = {'mae': MaskedMAE(compute_on_step=False),
               'mape': MaskedMAPE(compute_on_step=False),
               'mse': MaskedMSE(compute_on_step=False),
               'mre': MaskedMRE(compute_on_step=False)}

    # filler's inputs
    scheduler_class = CosineAnnealingLR if use_lr_schedule else None
    additional_filler_hparams = dict(model_class=model_cls,
                                     model_kwargs=model_kwargs,
                                     optim_class=torch.optim.Adam,
                                     optim_kwargs={'lr': lr,
                                                   'weight_decay': l2_reg},
                                     loss_fn=loss_fn,
                                     metrics=metrics,
                                     scheduler_class=scheduler_class,
                                     scheduler_kwargs={
                                         'eta_min': 0.0001,
                                         'T_max': epochs
                                     },
                                     alpha=alpha,
                                     hint_rate=hint_rate,
                                     g_train_freq=g_train_freq,
                                     d_train_freq=d_train_freq)

    filler_kwargs = parser_utils.filter_args(args={**split_conf, **additional_filler_hparams},
                                             target_cls=filler_cls,
                                             return_dict=True)

    filler = filler_cls(**filler_kwargs)

    ########################################
    # training                             #
    ########################################
    ########################################

    # callbacks
    early_stop_callback = EarlyStopping(monitor='val_mae', patience=patience, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1, monitor='val_mae', mode='min')


    logger = TensorBoardLogger(logdir, name="model")

    trainer = pl.Trainer(max_epochs=epochs,
                         logger=logger,
                         default_root_dir=logdir,
                         accelerator="gpu" if torch.cuda.is_available() else "cpu",  # Automatically detect GPU/CPU
                         devices=1 if torch.cuda.is_available() else "auto", # Use 1 GPU if available, otherwise default
                         gradient_clip_val=grad_clip_val,
                         gradient_clip_algorithm=grad_clip_algorithm,
                         callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(filler, datamodule=dm)

    ########################################
    # testing                              #
    ########################################

    filler.load_state_dict(
        torch.load(checkpoint_callback.best_model_path, lambda storage, loc: storage)[
            "state_dict"
        ]
    )
    filler.freeze()
    trainer.test(datamodule=dm)  # ✅ Explicitly passing the datamodule
    filler.eval()

    filler.eval()

    if torch.cuda.is_available():
        filler.cuda()

    with torch.no_grad():
        y_true, y_hat, mask = filler.predict_loader(dm.test_dataloader(), return_mask=True)

    # Debugging the shapes before reshaping
    print("🔍 y_hat shape before reshape:", y_hat.shape)
    print("🔍 Expected input_data shape:", input_data.shape)

    y_hat = y_hat.detach().cpu().numpy().reshape(input_data.shape)

    imputed_data = np.where(np.isnan(input), y_hat, input_data)  # Replace NaNs with predictions

    print("imputed_data.shape", imputed_data.shape)

    return imputed_data


if __name__ == '__main__':
    ts_1 = TimeSeries()

    # 2. load the timeseries from file or from the code
    ts_1.load_series(utils.search_path("eeg-alcohol"))  # shape 64x256
    ts_1.normalize(normalizer="min_max")

    # 3. contamination of the data
    ts_mask = ts_1.Contamination.missing_completely_at_random(ts_1.data)

    imputation, imputation_2 = recoveryGRIN(ts_mask)
