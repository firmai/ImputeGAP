import numpy as np
import torch
import tqdm

import imputegap
from imputegap.recovery.manager import TimeSeries
from imputegap.wrapper.AlgoPython.BayOTIDE import utils_BayOTIDE, model_BayOTIDE
import time
import warnings

warnings.filterwarnings("ignore")


def get_default_config(data_matrix=None):
    """
    Generate a default configuration for BayOTIDE with adaptive values
    based on input data.

    :param data_matrix: NumPy array (N x T) representing the dataset.
    :return: Dictionary containing hyperparameters.
    """
    # Use defaults if no data is provided
    N, T = (data_matrix.shape if data_matrix is not None else (64, 256))  # Default 64x256

    # Dynamically determine ranks based on data size
    K_trend = min(30, max(5, N // 3))  # Set trend rank (limit between 5 and N/3)
    K_season = min(3, max(1, T // 100))  # Adaptive seasonal rank (based on time steps)
    n_season = min(10, max(3, T // 50))  # Number of seasonal components
    K_bias = 1 if T > 100 else 0  # Include bias term if time series is long

    # Adaptive damping factors to avoid instability
    DAMPING_U = max(0.5, min(0.9, 1 - (N / (N + T))))  # Keep damping within (0.5, 0.9)
    DAMPING_tau = max(0.3, min(0.7, 1 - (T / (N + T))))  # Ensure reasonable updates
    DAMPING_W = (DAMPING_U + DAMPING_tau) / 2  # Keep W damping balanced

    # Avoid very small variance values that cause numerical issues
    a0 = max(1e-2, N / 100)  # Prior variance scaling
    b0 = max(1e-2, T / 100)  # Prior variance scaling
    v = max(0.5, min(5, N / T))  # Adaptive variance parameter

    # Auto-generate frequency list for K_season
    freq_list = [max(5, min(30, T // (10 + i * 5))) for i in range(K_season)]
    lengthscale_list = [max(0.01, min(0.1, T / (1000 + i * 100))) for i in range(K_season)]


    return {
        "INNER_ITER": 5,
        "EVALU_T": max(50, min(200, T // 10)),  # Adaptive evaluation interval
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_fold": 1,
        "time_scale": 1,
        "fix_int": True,
        "K_trend": K_trend,
        "K_season": K_season,
        "n_season": n_season,
        "K_bias": K_bias,
        "a0": a0,
        "b0": b0,
        "v": v,
        "THRE": 1.0e-4,
        "DAMPING_U": DAMPING_U,
        "DAMPING_tau": DAMPING_tau,
        "DAMPING_W": DAMPING_W,
        "kernel": {
            "kernel_trend": {
                "type": "Matern_21",
                "lengthscale": max(0.05, min(0.2, T / 500)),  # Adaptive lengthscale
                "variance": max(0.1, min(2.0, N / T)),  # Ensure variance is stable
                "noise": 1.0,
            },
            "kernel_season": {
                "type": "exp-periodic",
                "freq_list": freq_list,  # Now matches K_season
                "lengthscale_list": lengthscale_list,
                "noise": 1.0,
            }
        }
    }



def get_default_args():
    class Args:
        seed = 42
        num_fold = 1
        dataset = "default"
        task = "impute"
        r = 0.01
        other_para = ""

    return Args()


def recoveryBayOTIDE(data, K_trend=None, K_season=None, n_season=None, K_bias=None, time_scale=None, a0=None, b0=None, v=None, config=None, args=None):
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
    :param config: Dictionary containing hyperparameters (optional).
    :param args: Parsed arguments for the model (optional).
    :return: Imputed time series matrix (N x T).
    """
    data_matrix = data.copy()
    final_matrix = data.copy()
    missing_mask = np.isnan(data)


    # Replace NaN values in the dataset with 0
    data_matrix = np.nan_to_num(data_matrix, nan=0.0)

    # Load default configuration if not provided
    if config is None:
        config = get_default_config(data_matrix)

    # Override default configuration with user-provided values if not None
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

    if args is None:
        args = get_default_args()


    print("\n\n\t\t\t(PYTHON) BayOTIDE: Matrix Shape: (", data_matrix.shape[0], ", ", data_matrix.shape[1], ")")
    print(f"\t\t\tK_trend: {config['K_trend']}, K_season: {config['K_season']}, n_season: {config['n_season']}, "
          f"K_bias: {config['K_bias']}, time_scale: {config['time_scale']}, a0: {config['a0']}, "
          f"b0: {config['b0']}, v: {config['v']}")

    torch.random.manual_seed(args.seed)

    data_matrix = np.nan_to_num(data_matrix, nan=0.0)

    torch.random.manual_seed(args.seed)

    hyper_dict = utils_BayOTIDE.make_hyper_dict(config, args)

    INNER_ITER = hyper_dict["INNER_ITER"]
    EVALU_T = hyper_dict["EVALU_T"]


    for fold_id in range(args.num_fold):
        data_dict = utils_BayOTIDE.make_data_dict(hyper_dict, data_matrix, fold=fold_id)
        model = model_BayOTIDE.BayTIDE(hyper_dict, data_dict)
        model.reset()

        # One-pass along the time axis
        for T_id in tqdm.tqdm(range(model.T)):
            model.filter_predict(T_id)
            model.msg_llk_init()

            if model.mask_train[:, T_id].sum() > 0:
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
                print(
                    f"\t\t\t\t\t\tT_id = {T_id}, train_rmse = {loss_dict['train_RMSE']:.3f}, test_rmse= {loss_dict['test_RMSE']:.3f}")


        print('\t\t\t\tSmoothing back...')
        model.smooth()
        print('\t\t\t\tFinished training!')

        model.post_update_U_after_smooth(0)

        # **CRITICAL FIX: Ensure each series has unique imputation**
        W_matrix = model.post_W_m.clone().squeeze().cpu().detach().numpy()
        U_matrix = model.post_U_m.clone().squeeze().cpu().detach().numpy()

        print("\n\t\t\t\tW_matrix shape:", W_matrix.shape)  # Should be (N, K)
        print("\t\t\t\tU_matrix shape:", U_matrix.shape)  # Should be (K, T)

        # Check for row similarity
        w_unique_rows = np.unique(W_matrix, axis=0)
        print(f"\t\t\t\t\t\tUnique W_matrix rows: {w_unique_rows.shape[0]} / {W_matrix.shape[0]}")

        u_unique_rows = np.unique(U_matrix, axis=0)
        print(f"\t\t\t\t\t\tUnique W_matrix rows: {u_unique_rows.shape[0]} / {U_matrix.shape[0]}")

        # Ensure the multiplication preserves individual series variations
        imputed_matrix = np.matmul(W_matrix, U_matrix)  # (N, T)

        final_matrix[missing_mask] = imputed_matrix[missing_mask]

    return final_matrix


if __name__ == "__main__":
    ts_1 = TimeSeries()

    # 2. load the timeseries from file or from the code
    ts_1.load_series(imputegap.tools.utils.search_path("eeg-alcohol"))  # shape 64x256
    ts_1.normalize(normalizer="min_max")

    # 3. contamination of the data
    x = ts_1.Contamination.missing_percentage(ts_1.data, rate_series=0.4)

    imputation = recoveryBayOTIDE(x)