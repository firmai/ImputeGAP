########################################
## code copied and adapted from:      ##
## https://github.com/lmluzern/BRITS/ ##
########################################

import torch
import torch.optim as optim

import numpy as np

from . import utils
from . import models
from . import data_loader

from .data_prep_tf import prepare_dat

def train(model, input, epochs, batch_size):
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    data_iter = data_loader.get_loader(input, batch_size = batch_size)

    for epoch in range(0, epochs):
        model.train()

        run_loss = 0.0

        for idx, data in enumerate(data_iter):
            data = utils.to_var(data)
            ret = model.run_on_batch(data, optimizer)

            run_loss += ret['loss'].data

            print ('\t\t\t\r\t\t\t\t\t\tProgress epoch {}, {:.2f}%, average loss {}'.format(epoch, (idx + 1) * 100.0 / len(data_iter), run_loss / (idx + 1.0)))
    #end for    
    
    return (model, data_iter)
#end function

def evaluate(model, val_iter):
    model.eval()

    imputations = []

    for idx, data in enumerate(val_iter):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)
        
        imputation = ret['imputations'].data.cpu().numpy()
        imputations += imputation.tolist()
    #end for

    imputations = np.asarray(imputations)
    return imputations
#end function

def brits_recovery(incomp_data, model="brits", epoch=10, batch_size=7, nbr_features=1, hidden_layers=64, seq_length=36):
    matrix = incomp_data
    n = matrix.shape[1]
    prepare_dat(incomp_data, "incomp_data.tmp")

    if model != "brits_i_univ":
        print("\t\t(PYTHON) BRITS: Matrix Shape: (", incomp_data.shape[0], ", ", incomp_data.shape[1],
              ") for epoch ", epoch, ", batch_size ", batch_size, ", nbr features", nbr_features,
              ", seq_length ", seq_length, ", and hidden_layers ", hidden_layers, "...")

        model = getattr(models, model).Model(batch_size, nbr_features, hidden_layers, seq_length)
    else:
        print("\t\t(PYTHON) BRITS-UNIV: Matrix Shape: (", incomp_data.shape[0], ", ", incomp_data.shape[1],
              ") for epoch ", epoch, ", batch_size ", batch_size, ", nbr features", 1,
              ", seq_length ", n, ", and hidden_layers ", hidden_layers, "...")

        model = getattr(models, model).Model(batch_size, 1, hidden_layers, n)

    if torch.cuda.is_available():
        model = model.cuda()

    (model, data_iter) = train(model, "incomp_data.tmp", epoch, batch_size)
    res = evaluate(model, data_iter)

    recov = np.squeeze(np.array(res))
    print("recov", recov.shape)

    nan_mask = ~np.isnan(incomp_data)
    recov[nan_mask] = incomp_data[nan_mask]

    return recov


    #for i in range(0, len(res)):
    #    res_l = res[i, :n];
    #    matrix[:, i] = res_l.reshape(n);
    #end for
#end function
