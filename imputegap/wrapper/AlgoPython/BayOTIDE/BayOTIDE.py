# ===============================================================================================================
# SOURCE: https://github.com/xuangu-fang/BayOTIDE
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://arxiv.org/abs/2308.14906
# ===============================================================================================================


import os
import numpy as np
import torch 
import sys
import imputegap
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils
sys.path.append("../")
import tqdm
import yaml
torch.random.manual_seed(300)
from imputegap.wrapper.AlgoPython.BayOTIDE import utils_BayOTIDE, model_BayOTIDE
import time
import warnings
warnings.filterwarnings("ignore")
torch.random.manual_seed(300)


def generate_mask(data_matrix, drop_rate=0.8, valid_rate=0.2, verbose=False):
    """
    Generate train/test/valid masks based only on existing NaN positions in data_matrix.
    """
    nan_mask = np.isnan(data_matrix)
    nan_indices = np.argwhere(nan_mask)
    np.random.shuffle(nan_indices)

    n = len(nan_indices)
    n_test = int(n * drop_rate)
    n_valid = int(n * valid_rate)
    n_train = n - n_test - n_valid

    if verbose:
        print(f"\n{n =}")

    # Sanity check
    if verbose:
        print(f"\n{n_test =}")
        print(f"{n_valid =}")
        print(f"{n_train =}\n")

    train_idx = nan_indices[:n_train]
    test_idx = nan_indices[n_train:n_train + n_test]
    valid_idx = nan_indices[n_train + n_test:]

    if verbose:
        print(f"\n{train_idx.shape =}")
        print(f"{test_idx.shape =}")
        print(f"{valid_idx.shape =}\n")

    mask_train = np.zeros_like(data_matrix, dtype=np.uint8)
    mask_test = np.zeros_like(data_matrix, dtype=np.uint8)
    mask_valid = np.zeros_like(data_matrix, dtype=np.uint8)

    mask_train[tuple(train_idx.T)] = 1
    mask_test[tuple(test_idx.T)] = 1
    mask_valid[tuple(valid_idx.T)] = 1

    # Sanity check
    if verbose:
        print(f"\nTrain mask NaNs: {mask_train.sum()}")
        print(f"Test mask NaNs: {mask_test.sum()}")
        print(f"Valid mask NaNs: {mask_valid.sum()}")

        print(f"{mask_train.shape =}")
        print(f"{mask_test.shape =}")
        print(f"{mask_valid.shape =}\n")

    return mask_train, mask_test, mask_valid


def generate_random_mask(gt, mask_test, mask_valid, droprate=0.2, verbose=False, seed=300):
    """
    Generate a random training mask over the non-NaN entries of gt, excluding positions
    already present in the test and validation masks.

    Parameters
    ----------
    gt : numpy.ndarray
        Ground truth data (no NaNs).
    mask_test : numpy.ndarray
        Binary mask indicating test positions.
    mask_valid : numpy.ndarray
        Binary mask indicating validation positions.
    droprate : float
        Proportion of eligible entries to include in the training mask.
    verbose : bool
        Whether to print debug info.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Binary mask indicating training positions.
    """
    assert gt.shape == mask_test.shape == mask_valid.shape, "All input matrices must have the same shape"

    if seed is not None:
        np.random.seed(seed)

    # Valid positions: non-NaN and not in test/valid masks
    occupied_mask = (mask_test + mask_valid).astype(bool)
    eligible_mask = (~np.isnan(gt)) & (~occupied_mask)
    eligible_indices = np.argwhere(eligible_mask)

    n_train = int(len(eligible_indices) * droprate)

    np.random.shuffle(eligible_indices)
    selected_indices = eligible_indices[:n_train]

    mask_train = np.zeros_like(gt, dtype=np.uint8)
    mask_train[tuple(selected_indices.T)] = 1

    if verbose:
        print(f"\n4.d) test mask: {mask_test.sum()} values")
        print(f"4.d) valid mask: {mask_valid.sum()} values")
        print(f"4.d) eligible entries: {len(eligible_indices)}")
        print(f"4.d) selected training entries: {n_train}\n")

    # Sanity check: no overlap between training and test masks
    overlap = np.logical_and(mask_train, mask_test).sum()
    assert overlap == 0, f"Overlap detected between training and test masks: {overlap} entries."

    # Sanity check: no overlap between training and test masks
    overlap = np.logical_and(mask_train, mask_valid).sum()
    assert overlap == 0, f"Overlap detected between training and test masks: {overlap} entries."

    return mask_train




def generate_mask_imputegap(data_matrix, test_rate=0.8, valid_rate=0.2, verbose=False, seed=300):
    """
    Dispatch NaN positions in data_matrix to test and validation masks only.

    Parameters
    ----------
    data_matrix : numpy.ndarray
        Input matrix containing NaNs to be split.
    test_rate : float
        Proportion of NaNs to assign to the test set (default is 0.8).
    valid_rate : float
        Proportion of NaNs to assign to the validation set (default is 0.2).
        test_rate + valid_rate must equal 1.0.
    verbose : bool
        Whether to print debug info.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple
        test_mask : numpy.ndarray
            Binary mask indicating positions of NaNs in the test set.
        valid_mask : numpy.ndarray
            Binary mask indicating positions of NaNs in the validation set.
        n_nan : int
            Total number of NaN values found in the input matrix.
    """
    assert np.isclose(test_rate + valid_rate, 1.0), "test_rate and valid_rate must sum to 1.0"

    if seed is not None:
        np.random.seed(seed)

    nan_mask = np.isnan(data_matrix)
    nan_indices = np.argwhere(nan_mask)
    np.random.shuffle(nan_indices)

    n_nan = len(nan_indices)
    n_test = int(n_nan * test_rate)
    n_valid = n_nan - n_test

    if verbose:
        print(f"\n4.a) creating mask (testing, validation): Total NaNs = {n_nan}")
        print(f"4.b) creating mask: Assigned to test = {n_test}")
        print(f"4.c) creating mask: Assigned to valid = {n_valid}\n")

    test_idx = nan_indices[:n_test]
    valid_idx = nan_indices[n_test:]

    mask_test = np.zeros_like(data_matrix, dtype=np.uint8)
    mask_valid = np.zeros_like(data_matrix, dtype=np.uint8)

    mask_test[tuple(test_idx.T)] = 1
    mask_valid[tuple(valid_idx.T)] = 1

    return mask_test, mask_valid, n_nan



def prepare_testing_set(incomp_m, tr_ratio, seed=None):

    if seed is not None:
        np.random.seed(seed)

    data_matrix_cont = incomp_m.copy()

    target_ratio = 1 - tr_ratio
    total_values = data_matrix_cont.size
    target_n_nan = int(target_ratio * total_values)

    # 2) Current number of NaNs
    current_n_nan = np.isnan(data_matrix_cont).sum()
    n_new_nans = target_n_nan - current_n_nan

    # 2) Get available (non-NaN) indices as 2D coordinates
    available_indices = np.argwhere(~np.isnan(data_matrix_cont))

    # 3) Randomly pick indices to contaminate
    if n_new_nans > 0:
        chosen_indices = available_indices[:n_new_nans]
        for i, j in chosen_indices:
            data_matrix_cont[i, j] = np.nan

    # 4) check ratio
    n_total = data_matrix_cont.size
    n_nan = np.isnan(data_matrix_cont).sum()
    n_not_nan = n_total - n_nan

    # Compute actual ratios
    missing_ratio = n_nan / n_total
    observed_ratio = n_not_nan / n_total

    # Check if they match expectations (within a small tolerance)
    assert abs(missing_ratio - target_ratio) < 0.01, f"Missing ratio {missing_ratio} is not {target_ratio}"
    assert abs(observed_ratio - tr_ratio) < 0.01, f"Missing ratio {observed_ratio} is not {tr_ratio}"

    # Create the new mask
    new_mask = np.isnan(data_matrix_cont)

    return data_matrix_cont, new_mask


def recovBayOTIDE(incomp_m, K_trend=None, K_season=None, n_season=None, K_bias=None, time_scale=None, a0=None, b0=None, v=None, tr_ratio=0.6, config=None, args=None, verbose=True):
    """
    Run BayOTIDE model using a provided NumPy data matrix instead of loading from a file.

    :param data_matrix: Preloaded NumPy matrix containing time series data (N x T).
    :param K_trend: Number of trend factors (optional, overrides config if provided).
    :param K_season: Number of seasonal factors (optional, overrides config if provided).
    :param n_season: Number of seasonal components per factor (optional, overrides config if provided).
    :param K_bias: Number of bias factors (optional, overrides config if provided).
    :param time_scale: Scaling factor for the time step (optional, overrides config if provided).
    :param a0: Prior hyperparameter for variance scaling (optional, overrides config if provided).
    :param b0: Prior hyperparameter for variance scaling (optional, overrides config if provided).
    :param v: Variance hyperparameter for noise modeling (optional, overrides config if provided).
    :param tr_ratio: ratio of the training set.
    :param config: Dictionary containing hyperparameters (optional).
    :param args: Parsed arguments for the model (optional).
    :return: Imputed time series matrix (N x T).
    """

    gt_data_matrix = incomp_m.copy()
    cont_data_matrix = incomp_m.copy()
    final_result = incomp_m.copy()
    mask_original_nan = np.isnan(incomp_m)
    original_missing_ratio = utils.get_missing_ratio(cont_data_matrix)

    nan_replacement             = 0
    artificial_training_drop    = 0.4
    ts_ratio                    = 0.9
    val_ratio                   = 1-ts_ratio

    if verbose:
        print(f"\n1.a) original missing ratio = {original_missing_ratio:.2%}")
        print(f"1.b) original missing numbers = {np.sum(mask_original_nan)}")

    if abs((1-tr_ratio) - original_missing_ratio) > 0.01:
        cont_data_matrix, new_mask = prepare_testing_set(cont_data_matrix, tr_ratio)

        if verbose:
            print(f"1.c) building of the test set to reach a fix ratio of {1 - tr_ratio:.2%}...")
            final_ratio = utils.get_missing_ratio(cont_data_matrix)
            print(f"1.d) final artificially missing ratio for test set = {final_ratio:.2%}")
            print(f"1.e) final artificially missing numbers = {np.sum(new_mask)}\n")

    else:
        new_mask = mask_original_nan.copy()

    if verbose:
        print("\n2.a) reset all testing matrix values to 0 to prevent data leakage.")
    gt_data_matrix[new_mask] = nan_replacement
    assert not np.isnan(gt_data_matrix).any(), "gt_data_matrix still contains NaNs"
    assert (gt_data_matrix == nan_replacement).any(), "gt_data_matrix does not contain any zeros"

    sub_tensor = torch.from_numpy(gt_data_matrix).float()
    zero_ratio = (sub_tensor == nan_replacement).sum().item() / sub_tensor.numel()

    if verbose:
        print(f"2.b) check ratios : {zero_ratio = }")
        print(f"2.c) convert matrix to tensor (torch) : {sub_tensor.shape = }\n")

    if config is None:
        # Get directory of current file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "config_bayotide.yaml")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if verbose:
            print(f"\n3.a) loading of the configuration file : {config_path = }")

    # Conditional updates
    if K_trend is not None:
        config["K_trend"] = K_trend
    if K_season is not None:
        config["K_season"] = K_season
    if n_season is not None:
        config["n_season"] = n_season
    if K_bias is not None:
        config["K_bias"] = K_bias
    if time_scale is not None:
        config["time_scale"] = time_scale
    if a0 is not None:
        config["a0"] = a0
    if b0 is not None:
        config["b0"] = b0
    if v is not None:
        config["v"] = v

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"3.b) Using device: {device}\n")

    # Set device in config
    config["device"] = device
    seed = config["seed"]

    data_save = {}
    data_save['ndims'] = sub_tensor.shape
    data_save['raw_data'] = sub_tensor
    data_save['data'] = []
    data_save['time_uni'] = np.linspace(0, 1, sub_tensor.shape[1])

    for i in range(config["num_fold"]):
        mask_test, mask_valid, nbr_nans = generate_mask_imputegap(cont_data_matrix, test_rate=ts_ratio, valid_rate=val_ratio, verbose=verbose, seed=seed)
        mask_train = generate_random_mask(gt=gt_data_matrix, mask_test=mask_test, mask_valid=mask_valid, droprate=artificial_training_drop, verbose=verbose, seed=seed)
        data_save['data'].append({'mask_train': mask_train, 'mask_test': mask_test, 'mask_valid': mask_valid})

    if verbose:
        print(f"4.e) check : mask_test.sum() / mask_train.size = {mask_test.sum() / mask_train.size}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, config["data_path"])
    np.save(file_path, data_save)
    config["data_path"] = file_path

    if verbose:
        print(f"Data saved to {file_path}")

    data_file = file_path
    hyper_dict = config

    INNER_ITER = hyper_dict["INNER_ITER"]
    EVALU_T = hyper_dict["EVALU_T"]

    if verbose:
        print("\n\ntraining...")

    for fold_id in range(config["num_fold"]):

        data_dict = utils_BayOTIDE.make_data_dict(hyper_dict, data_file, fold=fold_id)

        model = model_BayOTIDE.BayTIDE(hyper_dict, data_dict)

        model.reset()

        # one-pass along the time axis
        for T_id in tqdm.tqdm(range(model.T)):
            model.filter_predict(T_id)
            model.msg_llk_init()

            if model.mask_train[:, T_id].sum() > 0:  # at least one obseved data at current step
                for inner_it in range(INNER_ITER):
                    flag = (inner_it == (INNER_ITER - 1))

                    model.msg_approx_U(T_id)
                    model.filter_update(T_id, flag)

                    model.msg_approx_W(T_id)
                    model.post_update_W(T_id)

                model.msg_approx_tau(T_id)
                model.post_update_tau(T_id)

            else:
                model.filter_update_fake(T_id)

            if T_id % EVALU_T == 0 or T_id == model.T - 1:
                _, loss_dict = model.model_test(T_id)

                if verbose:
                    print("T_id = {}, train_rmse = {:.3f}, test_rmse= {:.3f}".format(T_id, loss_dict["train_RMSE"], loss_dict["test_RMSE"]))

    if verbose:
        print('\n\n5.a) smoothing back...')

    model.smooth()
    model.post_update_U_after_smooth(0)

    # Run model test and get predictions
    pred, loss_dict = model.model_test(T_id)

    if verbose:
        print(f"{pred.shape =}")

    # Fill NaNs in original data
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()

    final_result[mask_original_nan] = pred[mask_original_nan]

    if verbose:
        print(f"{final_result.shape =}")

    return final_result


if __name__ == "__main__":

    ts = TimeSeries()
    ts.load_series(imputegap.tools.utils.search_path("chlorine"))
    print(f"{ts.data.shape = }")

    # contaminate the time series with MCAR pattern
    ts_m = ts.Contamination.mcar(ts.data)

    imputed_data = recovBayOTIDE(ts.data, ts_m)

    from imputegap.recovery.imputation import Imputation

    imputer = Imputation.DeepLearning.BayOTIDE(ts_m)
    imputer.recov_data = imputed_data
    imputer.incomp_data = ts_m

    # compute and print the imputation metrics
    imputer.score(ts.data, imputed_data)
    ts.print_results(imputer.metrics)

    # plot the recovered time series
    ts.plot(input_data=ts.data, incomp_data=ts_m, recov_data=imputed_data, nbr_series=9, subplot=True, algorithm=imputer.algorithm, save_path="./imputegap_assets/imputation")
    ts.plot(input_data=ts.data, nbr_series=9,  save_path="./imputegap_assets/imputation")
    ts.plot(input_data=imputed_data, nbr_series=9,  save_path="./imputegap_assets/imputation")