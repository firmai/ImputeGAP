import torch
import numpy as np

from imputegap.tools import utils
from imputegap.wrapper.AlgoPython.priSTI.models.pristi import PriSTI_
from imputegap.wrapper.AlgoPython.priSTI.utils import train, evaluate
from imputegap.wrapper.AlgoPython.priSTI import get_dataloader


def recovPRISTI(data, target_strategy="hybrid", unconditional=True, seed=42, device="cpu", num_workers=1):

    print("\t\t(PYTHON) priSTI: Matrix Shape: (", data.shape[0], ", ", data.shape[1], ") for target_strategy ",
          target_strategy, ", unconditional ", unconditional, ", and device ", device, "(num_workers", num_workers,")...")

    n, dim = data.shape
    SEED = seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    config = utils.load_parameters(query="default", algorithm="other")

    config["model"]["is_unconditional"] = int(unconditional)
    config["model"]["target_strategy"] = target_strategy
    config["diffusion"]["adj_file"] = None
    config["seed"] = SEED

    # print(json.dumps(config, indent=4))

    data_loader = get_dataloader.data_loader(data, is_interpolate=config["model"]["use_guide"],
        num_workers=num_workers, target_strategy=target_strategy, mask_sensor=config["model"]["mask_sensor"],
        shuffle=False
    )

    config["train"]["batch_size"] = data.shape[1]

    model = PriSTI_(config, device, target_dim=dim, seq_len=n).to(device)
    print("\t\t\tStarting training\n")
    train(
        model,
        config["train"],
        data_loader,
        valid_loader=data_loader
    )
    
    print("\t\t\tStarting evaluation\n")
    matrix = evaluate(model, data_loader, nsample=1)

    return np.array(matrix)