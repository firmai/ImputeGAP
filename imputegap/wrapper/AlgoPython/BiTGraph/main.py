import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import imputegap
from imputegap.recovery.manager import TimeSeries
from imputegap.wrapper.AlgoPython.BiTGraph.data.GenerateDataset import loaddataset

from imputegap.wrapper.AlgoPython.BiTGraph.models.BiaTCGNet.BiaTCGNet import Model

torch.multiprocessing.set_sharing_strategy('file_system')

import argparse

if os.getenv("GITHUB_ACTIONS") == "true" or not torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cuda")

args = argparse.Namespace(
    epochs=None,
    batch_size=1,
    task='prediction',
    adj_threshold=0.1,
    dataset='imputegap',
    val_ratio=0.2,
    test_ratio=0.2,
    column_wise=False,
    seed=-1,
    precision=32,
    model_name='spin',
    dataset_name='imputegap-set',
    fc_dropout=0.2,
    head_dropout=0,
    individual=0,
    patch_len=8,
    padding_patch='end',
    revin=0,
    affine=0,
    subtract_last=0,
    decomposition=0,
    kernel_size=25,
    kernel_set=[2, 3, 6, 7],
    enc_in=1,
    dec_in=1,
    c_out=1,
    d_model=512,
    n_heads=8,
    e_layers=2,
    d_layers=2,
    d_ff=2048,
    moving_avg=[24],
    factor=1,
    dropout=0.05,
    embed='timeF',
    activation='gelu',
    freq='h',
    num_nodes=1,
    version='Fourier',
    mode_select='random',
    modes=64,
    L=3,
    base='legendre',
    cross_activation='tanh',
    input_dim=1,
    output_dim=1,
    embed_dim=512,
    rnn_units=64,
    num_layers=2,
    cheb_k=2,
    default_graph=True,
    temperature=0.5,
    config_filename='',
    config='imputation/spin.yaml',
    output_attention=False,
    val_len=0.2,
    test_len=0.2,
    mask_ratio=0.1,
    lr=0.001,
    patience=40,
    l2_reg=0.0,
    batch_inference=32,
    split_batch_in=1,
    grad_clip_val=5.0,
    loss_fn='l1_loss',
    lr_scheduler=None,
    seq_len=24,
    label_len=12,
    pred_len=24,
    horizon=24,
    delay=0,
    stride=1,
    window_lag=1,
    horizon_lag=1
)


criteron = nn.L1Loss().to(device)

if(args.dataset=='Metr'):
    node_number=207
    args.num_nodes=207
    args.enc_in=207
    args.dec_in=207
    args.c_out=207
elif(args.dataset=='PEMS'):
    node_number=325
    args.num_nodes=325
    args.enc_in = 325
    args.dec_in = 325
    args.c_out = 325
elif(args.dataset=='ETTh1'):
    node_number=7
    args.num_nodes=7
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
elif(args.dataset=='Elec'):
    node_number=321
    args.num_nodes=321
    args.enc_in = 321
    args.dec_in = 321
    args.c_out = 321
elif(args.dataset=='BeijingAir'):
    node_number=36
    args.num_nodes=36
    args.enc_in = 36
    args.dec_in = 36
    args.c_out = 36

def train(model, data=None):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.seed > 0:
        args.seed = np.random.randint(args.seed)
    torch.set_num_threads(1)

    train_dataloader, val_dataloader, test_dataloader, recov_dataloader, scaler = loaddataset(args.seq_len, args.pred_len, args.mask_ratio, args.dataset, args.batch_size, data)

    best_loss=9999999.99
    k=0
    for epoch in range(args.epochs):
        model.train()

        for i, (x, y, mask, target_mask) in enumerate(train_dataloader):
            x, y, mask,target_mask = x.to(device), y.to(device), mask.to(device), target_mask.to(device)
            x=x*mask
            y=y*target_mask
            x_hat=model(x,mask,k)
            loss = torch.sum(torch.abs(x_hat-y)*target_mask)/torch.sum(target_mask)
            optimizer.zero_grad()  # optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # Concatenate all batches

        loss, _ =evaluate(model, val_dataloader, scaler)
        print('\t\t\t\t\t\tepoch, loss:',epoch,loss)

        if(loss<best_loss):
            best_loss=loss
            best_model = copy.deepcopy(model.state_dict())
            _, imputed_matrix = evaluate(model, recov_dataloader, scaler)

            print('\t\t\t\t\t\t\t\tbest_loss:', best_loss)



    return imputed_matrix, best_model    # Return the imputed matrix



def evaluate(model, val_iter, scaler):
    model.eval()
    loss=0.0
    k=0
    imputed_results = []  # Store imputed batches

    with torch.no_grad():
        for i, (x,y,mask,target_mask) in enumerate(val_iter):
            x, y, mask, target_mask = x.to(device), y.to(device), mask.to(device), target_mask.to(device)

            x_hat=model(x,mask,k)

            x_hat = scaler.inverse_transform(x_hat)
            y = scaler.inverse_transform(y)

            losses = torch.sum(torch.abs(x_hat-y)*target_mask)/torch.sum(target_mask)
            loss+=losses

            imputed_results.append(x_hat.detach().cpu().numpy())  # Store the batch imputation

    imputed_results.append(x_hat.detach().cpu().numpy())  # Store the batch imputation
    full_imputed_matrix = np.concatenate(imputed_results, axis=0)

    return loss/len(val_iter), np.array(full_imputed_matrix)



def recoveryBitGRAPH(input=None, node_number=-1, kernel_set=[1], dropout=0.3, subgraph_size=5, node_dim=3, seq_len=1, lr=0.001, epoch=3, seed=42):

    data = np.copy(input)
    recov = np.copy(input)
    missing_mask = np.isnan(input)

    args.kernel_set= kernel_set
    args.dropout=dropout
    args.subgraph_size=subgraph_size
    args.node_dim=node_dim
    args.seq_len=seq_len
    args.lr=lr
    args.epochs=epoch
    args.seed=seed

    if node_number==-1 and data is not None:
        node_number = data.shape[1]
        args.pred_len = 1


    print("\n\n\t\t\t(PYTHON) BitGRAPH: Matrix Shape: (", data.shape[0], ", ", data.shape[1], ")")
    print(f"\t\t\tnode_number: {node_number}, kernel_set: {args.kernel_set}, dropout: {args.dropout}, "
          f"subgraph_size: {args.subgraph_size}, node_dim: {args.node_dim}, seq_len: {args.seq_len}, "
          f"lr: {args.lr}, epochs: {args.epochs}, pred_len: {args.pred_len}, and seed {args.seed}")

    model=Model(True, True, 2, node_number,args.kernel_set, device=device.type,
                predefined_A=None, dropout=args.dropout, subgraph_size=args.subgraph_size, node_dim=args.node_dim,
                dilation_exponential=1, conv_channels=8, residual_channels=8, skip_channels=16, end_channels= 32,
                seq_length=args.seq_len, in_dim=1,out_len=args.pred_len, out_dim=1, layers=2, propalpha=0.05,
                tanhalpha=3, layer_norm_affline=True) #2 4 6

    model.to(device)

    imputed_matrix, best_model = train(model, data=data)

    print("\t\t\t\t\t\tImputed Matrix Shape Before Reshaping:", imputed_matrix.shape)

    # Ensure proper reshaping to (64, 256)
    reshaped_imputed_matrix = imputed_matrix.reshape(data.shape[0], data.shape[1])

    print("\t\t\t\t\t\tReshaped Imputed Matrix Shape:", reshaped_imputed_matrix.shape)

    final_imputed_matrix = np.where(missing_mask, reshaped_imputed_matrix, recov)

    return final_imputed_matrix



if __name__ == '__main__':

    ts_1 = TimeSeries()
    ts_1.load_series(imputegap.tools.utils.search_path("eeg-alcohol"))  # shape (64, 256)
    ts_1.normalize(normalizer="min_max")

    ts_mask = ts_1.Contamination.missing_completely_at_random(ts_1.data, rate_dataset=0.4)

    imputed_matrix = recoveryBitGRAPH(ts_mask)
