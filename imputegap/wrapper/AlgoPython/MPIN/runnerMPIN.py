import pandas as pd
import torch

from imputegap.wrapper.AlgoPython.MPIN.utils.load_dataset import get_model_size
from imputegap.wrapper.AlgoPython.MPIN.utils.regressor import MLPNet
from imputegap.wrapper.AlgoPython.MPIN.utils.DynamicGNN import DynamicGCN, DynamicGAT, DynamicGraphSAGE, StaticGCN, \
    StaticGraphSAGE, StaticGAT

import torch.optim as optim
import copy
import numpy as np
import random

from torch_geometric.nn import knn_graph

def data_transform(X, X_mask, eval_ratio=0.05):
    eval_mask = np.zeros(X_mask.shape)
    rows, cols = np.where(X_mask == 1)
    eval_row_index_index = random.sample(range(len(rows)), int(eval_ratio * len(rows)))
    eval_row_index = rows[eval_row_index_index]
    eval_col_index = cols[eval_row_index_index]
    X_mask[eval_row_index, eval_col_index] = 0
    eval_mask[eval_row_index, eval_col_index] = 1
    eval_X = copy.copy(X)
    X[eval_row_index, eval_col_index] = 0
    return X, X_mask, eval_X, eval_mask


def build_GNN(in_channels, out_channels, k, base, device):
    if base == 'GAT':
        gnn = DynamicGAT(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
    elif base == 'GCN':
        gnn = DynamicGCN(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
    elif base == 'SAGE':
        gnn = DynamicGraphSAGE(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
    return gnn


def build_GNN_static(in_channels, out_channels, k, base, device):
    if base == 'GAT':
        gnn = StaticGAT(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
    elif base == 'GCN':
        gnn = StaticGCN(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
    elif base == 'SAGE':
        gnn = StaticGraphSAGE(in_channels=in_channels, out_channels=out_channels, k=k).to(device)
    return gnn


def get_window_data(base_X, base_X_mask, start, end, ratio):
    X = base_X[int(len(base_X) * start * ratio):int(len(base_X) * end * ratio)]
    X_mask = base_X_mask[int(len(base_X) * start * ratio):int(len(base_X) * end * ratio)]
    return X, X_mask


def window_imputation(input, mask, start, end, sample_ratio, initial_state_dict=None, X_last=None, mask_last=None,
                      mae_last=None, transfer=False, lr=0.01, weight_decay=0.1, epochs=200, out_channels=256,
                      state=True, k=10, base="SAGE", thre=0.25, eval_ratio=0.05, dynamic=False, device=None):

    X, X_mask = get_window_data(base_X=input, base_X_mask=mask, start=start, end=end, ratio=sample_ratio)

    print("\n\n\t\t\t(PYTHON) window_imputation: Matrix Shape: (", input.shape[0], ", ", input.shape[1], ") for",
          " k ", k, " lr ", lr, " weight ", weight_decay, " epochs ", epochs, " threshold ", thre,
          ", and base ", base, "=================================================\n\n ")

    ori_X = copy.copy(X)
    feature_dim = ori_X.shape[1]
    ori_X_mask = copy.copy(X_mask)

    all_mask = copy.copy(X_mask)
    all_X = copy.copy(X)

    if X_last:
        X_last = np.array(X_last)
        all_X = np.concatenate([X_last, X], axis=0)
        all_mask = np.concatenate([mask_last, X_mask], axis=0)

        X_last = X_last.tolist()

    all_mask_ts = torch.FloatTensor(all_mask).to(device)

    gram_matrix = torch.mm(all_mask_ts, all_mask_ts.t())  # compute the gram product

    gram_vec = gram_matrix.diagonal()

    gram_row_sum = gram_matrix.sum(dim=0)

    value_vec = gram_vec - (gram_row_sum - gram_vec) / (gram_matrix.shape[0] - 1)

    keep_index = torch.where(value_vec > thre * (feature_dim - 1))[0]
    keep_index = keep_index.data.cpu().numpy()

    keep_mask = all_mask[keep_index]

    keep_X = all_X[keep_index]

    X, X_mask, eval_X, eval_mask = data_transform(X, X_mask, eval_ratio=eval_ratio)

    if X_last:
        X_last = np.array(X_last)
        shp_last = X_last.shape
        eval_X = np.concatenate([X_last, eval_X], axis=0)
        X = np.concatenate([X_last, X], axis=0)
        eval_mask_last = np.zeros(shp_last)
        eval_mask = np.concatenate([eval_mask_last, eval_mask], axis=0)
        X_mask = np.concatenate([mask_last, X_mask], axis=0)

    in_channels = X.shape[1]
    out_channels = X.shape[0]
    X = torch.FloatTensor(X).to(device)
    X_mask = torch.LongTensor(X_mask).to(device)
    eval_X = torch.FloatTensor(eval_X).to(device)
    eval_mask = torch.LongTensor(eval_mask).to(device)

    # build model
    if dynamic == 'true':
        gnn = build_GNN(in_channels=in_channels, out_channels=out_channels, k=k, base=base, device=device)
        gnn2 = build_GNN(in_channels=in_channels, out_channels=out_channels, k=k, base=base, device=device)
    else:
        gnn = build_GNN_static(in_channels=in_channels, out_channels=out_channels, k=k, base=base, device=device)
        gnn2 = build_GNN_static(in_channels=in_channels, out_channels=out_channels, k=k, base=base, device=device)

    model_list = [gnn, gnn2]
    regressor = MLPNet(out_channels, in_channels).to(device)

    if initial_state_dict != None:
        gnn.load_state_dict(initial_state_dict['gnn'])
        gnn2.load_state_dict(initial_state_dict['gnn2'])
        if not transfer:
            regressor.load_state_dict(initial_state_dict['regressor'])

    trainable_parameters = []
    for model in model_list:
        trainable_parameters.extend(list(model.parameters()))

    trainable_parameters.extend(list(regressor.parameters()))
    filter_fn = list(filter(lambda p: p.requires_grad, trainable_parameters))

    num_of_params = sum(p.numel() for p in filter_fn)

    model_size = get_model_size(gnn) + get_model_size(gnn2) + get_model_size(regressor)
    model_size = round(model_size, 6)

    num_of_params = num_of_params / 1e6

    opt = optim.Adam(filter_fn, lr=lr, weight_decay=weight_decay)

    graph_impute_layers = len(model_list)

    X_knn = copy.deepcopy(X)

    edge_index = knn_graph(X_knn, k, batch=None, loop=False, cosine=False)

    min_mae_error = 1e9
    min_mse_error = None
    min_mape_error = None
    opt_epoch = None
    opt_time = None
    best_X_imputed = None
    best_state_dict = None

    for pre_epoch in range(epochs):
        gnn.train()
        gnn2.train()
        regressor.train()
        opt.zero_grad()
        loss = 0
        X_imputed = copy.copy(X)

        for i in range(graph_impute_layers):
            if dynamic == 'true':
                X_emb = model_list[i](X_imputed)
            else:
                X_emb, edge_index = model_list[i](X_imputed, edge_index)

            pred = regressor(X_emb)
            X_imputed = X * X_mask + pred * (1 - X_mask)
            temp_loss = torch.sum(torch.abs(X - pred) * X_mask) / (torch.sum(X_mask) + 1e-5)
            # print('temp loss:', temp_loss.item())
            loss += temp_loss

        loss.backward()
        opt.step()
        train_loss = loss.item()
        print('\n\t\t\t{n} epoch loss:'.format(n=pre_epoch), train_loss, '.............')

        trans_X = copy.copy(X_imputed)
        trans_eval_X = copy.copy(eval_X)

        epoch_state_dict = {'gnn': gnn.state_dict(), 'gnn2': gnn2.state_dict(), 'regressor': regressor.state_dict()}

        gnn.eval()
        gnn2.eval()
        regressor.eval()

        with torch.no_grad():

            mae_error = torch.sum(torch.abs(trans_X - trans_eval_X) * eval_mask) / torch.sum(eval_mask)
            mse_error = torch.sum(((trans_X - trans_eval_X) ** 2) * eval_mask) / torch.sum(eval_mask)
            mape_error = torch.sum(torch.abs(trans_X - trans_eval_X) * eval_mask ) / (torch.sum(torch.abs(trans_eval_X) * eval_mask) + 1e-12)
            print('\t\t\t\timputegap impute error MAE:', mae_error.item())
            print('\t\t\t\timputegap impute error MSE:', mse_error.item())
            print('\t\t\t\timputegap impute error MRE:', mape_error.item())


            if mae_error.item() < min_mae_error:
                opt_epoch = copy.copy(pre_epoch)
                min_mae_error = round(mae_error.item(), 6)
                print('\t\t\t{epoch}_opt_mae_error'.format(epoch=pre_epoch), min_mae_error)

                min_mse_error = round(mse_error.item(), 6)
                min_mape_error = round(mape_error.item(), 6)

                print('\t\t\t{epoch}_opt time:'.format(epoch=pre_epoch), opt_time)

                best_X_imputed = copy.copy(X_imputed)
                best_X_imputed = best_X_imputed * (1 - ori_X_mask) + ori_X * ori_X_mask
                best_state_dict = copy.copy(epoch_state_dict)

    results_list = [opt_epoch, min_mae_error, min_mse_error, min_mape_error, num_of_params, model_size, opt_time, 0]

    if mae_last and (min_mae_error > mae_last) and (state == 'true'):
        best_state_dict = copy.copy(initial_state_dict)
    return best_state_dict, keep_X.tolist(), keep_mask, results_list, min_mae_error, best_X_imputed


def recoverMPIN(input, mode="alone", window=2, k=10, lr=0.01, weight_decay=0.1, epochs=200, num_of_iteration=5, thre=0.25,
                base="SAGE", out_channels=64, eval_ratio=0.05, state=True, dynamic=True, seed=0):

    print("\t\t(PYTHON) MPIN: Matrix Shape: (", input.shape[0], ", ", input.shape[1], ") for mode ", mode,
          ", window ", window, " k ", k, " lr ", lr, " weight ", weight_decay, " epochs ", epochs, " num_of_iteration ",
          num_of_iteration, " threshold ", thre, ", and base ", base, "...")

    torch.random.manual_seed(seed)
    device = torch.device('cpu')

    random.seed(seed)
    base_X = input
    base_X_mask = (~np.isnan(base_X)).astype(int)
    base_X = np.nan_to_num(base_X)

    num_windows = window

    results_schema = ['opt_epoch', 'opt_mae', 'mse', 'mape', 'para', 'memo', 'opt_time', 'tot_time']

    iter_results_list = []
    best_X_imputed = 0

    for iteration in range(num_of_iteration):
        results_collect = []
        for w in range(num_windows):
            print(f'\t\t\twhich time window:{w}')
            if w == 0:
                window_best_state, X_last, mask_last, window_results, mae_last, best_X = window_imputation(input=base_X,
                                                                                                           mask=base_X_mask,
                                                                                                           start=w,
                                                                                                           end=w + 1,
                                                                                                           sample_ratio=1 / num_windows,
                                                                                                           lr=lr,
                                                                                                           weight_decay=weight_decay,
                                                                                                           epochs=epochs,
                                                                                                           out_channels=out_channels,
                                                                                                           state=state,
                                                                                                           k=k,
                                                                                                           base=base,
                                                                                                           thre=thre,
                                                                                                           eval_ratio=eval_ratio,
                                                                                                           dynamic=dynamic,
                                                                                                           device=device)
                results_collect.append(window_results)
                best_X_imputed = best_X
            else:
                if mode == 'alone':
                    window_best_state, X_last, mask_last, window_results, mae_last, best_X = window_imputation(
                        input=base_X, mask=base_X_mask, start=w, end=w + 1, sample_ratio=1 / num_windows, lr=lr,
                        weight_decay=weight_decay, epochs=epochs, out_channels=out_channels, state=state, k=k,
                        base=base, thre=thre, eval_ratio=eval_ratio, dynamic=dynamic, device=device)
                    results_collect.append(window_results)
                    best_X_imputed = best_X

                elif mode == 'data':
                    window_best_state, X_last, mask_last, window_results, mae_last, best_X = window_imputation(
                        input=base_X, mask=base_X_mask, start=w, end=w + 1, sample_ratio=1 / num_windows, X_last=X_last,
                        mask_last=mask_last, lr=lr, weight_decay=weight_decay, epochs=epochs, out_channels=out_channels,
                        state=state, k=k, base=base, thre=thre, eval_ratio=eval_ratio, dynamic=dynamic, device=device)
                    results_collect.append(window_results)
                    best_X_imputed = best_X

                elif mode == 'state':
                    window_best_state, X_last, mask_last, window_results, mae_last, best_X = window_imputation(
                        input=base_X, mask=base_X_mask, start=w, end=w + 1, sample_ratio=1 / num_windows,
                        initial_state_dict=window_best_state, mae_last=mae_last, lr=lr, weight_decay=weight_decay,
                        epochs=epochs, out_channels=out_channels, state=state, k=k, base=base, thre=thre,
                        eval_ratio=eval_ratio, dynamic=dynamic, device=device)
                    results_collect.append(window_results)
                    best_X_imputed = best_X

                elif mode == 'state+transfer':
                    window_best_state, X_last, mask_last, window_results, mae_last, best_X = window_imputation(
                        input=base_X, mask=base_X_mask, start=w, end=w + 1, sample_ratio=1 / num_windows,
                        initial_state_dict=window_best_state, transfer=True, mae_last=mae_last, lr=lr,
                        weight_decay=weight_decay, epochs=epochs, out_channels=out_channels, state=state, k=k,
                        base=base, thre=thre, eval_ratio=eval_ratio, dynamic=dynamic, device=device)
                    results_collect.append(window_results)
                    best_X_imputed = best_X

                elif mode == 'data+state':
                    window_best_state, X_last, mask_last, window_results, mae_last, best_X = window_imputation(
                        input=base_X, mask=base_X_mask, start=w, end=w + 1, sample_ratio=1 / num_windows,
                        initial_state_dict=window_best_state, X_last=X_last, mask_last=mask_last, mae_last=mae_last,
                        lr=lr, weight_decay=weight_decay, epochs=epochs, out_channels=out_channels, state=state, k=k,
                        base=base, thre=thre, eval_ratio=eval_ratio, dynamic=dynamic, device=device)
                    results_collect.append(window_results)
                    best_X_imputed = best_X

                elif mode == 'data+state+transfer':
                    window_best_state, X_last, mask_last, window_results, mae_last, best_X = window_imputation(
                        input=base_X, mask=base_X_mask, start=w, end=w + 1, sample_ratio=1 / num_windows,
                        initial_state_dict=window_best_state, X_last=X_last, mask_last=mask_last, transfer=True,
                        mae_last=mae_last, lr=lr, weight_decay=weight_decay, epochs=epochs, out_channels=out_channels,
                        state=state, k=k, base=base, thre=thre, eval_ratio=eval_ratio, dynamic=dynamic, device=device)
                    results_collect.append(window_results)
                    best_X_imputed = best_X

        df = pd.DataFrame(results_collect, index=range(num_windows), columns=results_schema)
        iter_results_list.append(df)
        # print(res.shape)
    print('\t\t\tdone!')

    avg_df = sum(iter_results_list) / num_of_iteration
    avg_df = avg_df.round(4)

    best_X_imputed = np.array(best_X_imputed)
    nan_mask = ~np.isnan(input)
    best_X_imputed[nan_mask] = input[nan_mask]

    return best_X_imputed
