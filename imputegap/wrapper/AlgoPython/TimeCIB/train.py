import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm
from types import SimpleNamespace

import imputegap
from imputegap.recovery.manager import TimeSeries
from imputegap.wrapper.AlgoPython.TimeCIB.models import TimeCIB, RNNEncoder, GaussianDecoder, DiagonalEncoder, \
    JointEncoder, BandedJointEncoder, ImagePreprocessor
from imputegap.wrapper.AlgoPython.TimeCIB.models.utils import moving_avg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)


def recoveryTimeCIB(miss_np_data):
    """
    Main function adapted to take preloaded NumPy data.
    """
    # Define default arguments using SimpleNamespace
    args = SimpleNamespace(
        projname="custom_project",
        runname="custom_run",
        dataset="custom",
        model_type="timecib",
        prior_type="norm",
        encoder_type="rnn",
        lamda=1.0,
        beta=1.0,
        latent_dim=32,
        sim_type="cauchy",
        dir="",
        cont_length_scale=2.0,
        cont_period_scale=24.0,
        cont_conf=False,
        imputed=None,
        missingtype="mnar",
        missingratio=None,
        temperature=1.0,
        num_epoch=30,
        seed=0,
        test=False,
        weight_decay=1e-5,
        gradient_clip=1e5,
        print_interval=1000,
        return_parts=True,
        device=device,
        cnn_kernel_size=3,
        cnn_sizes=[256],
        testing=False,
        kernel="cauchy",
        kernel_scales=1,
        normalize=True,
        cont_sigma=1.0,
        batch_size=1,  # 64 (number of sequences)
        data_dim=miss_np_data.shape[1],  # 256 (time length)
        input_dim=miss_np_data.shape[1],  # ✅ Correct: Only 1 feature per time step
        time_length=miss_np_data.shape[0],  # 256
        binary=False,
        num_classes=0,
        image_shape=None,
        encoder_sizes=[128, 128],
        decoder_sizes=[256, 256],
        window_size=1,
        sigma=1.005,
        length_scale=2.0,
        learning_rate=1e-3,
        transpose=False,
        extractor="cnn",
        image_preprocessor=None
    )

    # Fix seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)

    # Define encoder and decoder explicitly
    encoders = {
        "diag": DiagonalEncoder,
        "joint": JointEncoder,
        "band": BandedJointEncoder,
        "rnn": RNNEncoder
    }
    args.encoder = encoders.get(args.encoder_type, RNNEncoder)
    args.decoder = GaussianDecoder  # Ensure decoder is set correctly

    # Prepare dataset directly from `miss_np_data`
    imputegap_data = torch.tensor(miss_np_data, dtype=torch.float32).to(device)  # (64, 256)

    #############
    # Fix seeds #
    #############
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)

    #############
    # Load data #
    #############
    from dataset import UnifiedDataset
    train_dataset = UnifiedDataset(train=True, test=False, args=args, data=imputegap_data)
    test_dataset = UnifiedDataset(train=False, test=True, args=args, data=imputegap_data)

    ################
    # Build models #
    ################

    if args.encoder_type == "diag":
        args.encoder = getattr(args, "encoder", DiagonalEncoder)
    elif args.encoder_type == "joint":
        args.encoder = getattr(args, "encoder", JointEncoder)
    elif args.encoder_type == "band":
        args.encoder = getattr(args, "encoder", BandedJointEncoder)
    elif args.encoder_type == "rnn":
        args.encoder = getattr(args, "encoder", RNNEncoder)
    else:
        raise ValueError("Encoder type must be one of ['diag', 'joint', 'band', 'rnn']")

    model = TimeCIB(args)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    ##################
    # Start training #
    ##################
    best_valid_mse = 1e+3
    best_valid_renew = False
    for epoch in range(args.num_epoch):
        if not args.test:
            print("--------------------")
            print(f"Current Epoch: {epoch + 1} / {args.num_epoch}")

            if epoch == 0:
                idx, batch = next(enumerate(test_dataset.loader))
                x_full, x_miss, m_miss, m_artificial, y, t = batch
                x_miss, m_miss, t = x_miss.to(device), m_miss.to(device), t.to(device)
                (x_z, x_z_nott) = model.sample(x_miss, t, m_miss)
                x_z, x_z_nott = x_z.detach().cpu().numpy(), x_z_nott.detach().cpu().numpy()

            ### Train
            model.train()
            train_loss, train_nll, train_kl, train_smi, train_mae, train_mse, train_rmse, train_mre, train_nll, train_mnll, train_auroc, num_samples, num_missing = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

            for idx, batch in enumerate(tqdm(train_dataset.loader)):
                x_full, x_miss, m_miss, m_artificial, y, t = batch

                batch_size = len(x_full)
                num_samples += batch_size
                curr_missing = torch.sum(m_artificial).item()
                num_missing += curr_missing

                x_miss, x_full, m_miss, m_artificial, t = x_miss.to(device), x_full.to(device), m_miss.to(
                    device), m_artificial.to(device), t.to(device)
                loss, nll, kl, smi, mae, mse, rmse, mre, mnll = model(x_miss, x_full, m_miss, m_artificial, t)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()

                train_loss, train_nll, train_kl, train_smi, train_mae, train_mse, train_rmse, train_mre, train_mnll = moving_avg(
                    loss, nll, kl, smi, mae, mse, rmse, mre, mnll, train_loss, train_nll, train_kl, train_smi,
                    train_mae, train_mse, train_rmse, train_mre, train_mnll, num_samples, batch_size, num_missing,
                    curr_missing)

                if idx % args.print_interval == 0 and idx > 0:
                    if args.num_classes > 0:
                        print(f"Train step {idx} | Loss: {train_loss:.3f} | NLL:{train_nll:.3f} | KL:{train_kl:.3f} | SMI:{train_smi:.1f} | MAE:{train_mae:.4f} | MSE:{train_mse:.4f} | RMSE:{train_rmse:.4f} | MRE:{train_mre:.4f} | MNLL:{train_mnll:.4f}")
                    else:
                        print(f"Train step {idx} | Loss: {train_loss:.3f} | NLL:{train_nll:.3f} | KL:{train_kl:.3f} | SMI:{train_smi:.1f} | MAE:{train_mae:.4f} | MSE:{train_mse:.4f} | RMSE:{train_rmse:.4f} | MRE:{train_mre:.4f} | MNLL:{train_mnll:.4f}")

            print(f"Train step {idx} | Loss: {train_loss:.3f} | NLL:{train_nll:.3f} | KL:{train_kl:.3f} | SMI:{train_smi:.1f} | MAE:{train_mae:.4f} | MSE:{train_mse:.4f} | RMSE:{train_rmse:.4f} | MRE:{train_mre:.4f} | MNLL:{train_mnll:.4f}")


    # Generate imputed matrix
    model.eval()

    print("Imputed Matrix:")

    return None


if __name__ == "__main__":
    ts_1 = TimeSeries()
    ts_1.load_series(imputegap.tools.utils.search_path("eeg-alcohol"))  # shape (64, 256)
    ts_1.normalize(normalizer="min_max")

    miss_data = ts_1.Contamination.missing_completely_at_random(ts_1.data, rate_series=0.4)

    recoveryTimeCIB(miss_data)
