from __future__ import print_function

import random
import time
import warnings

from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils
from types import SimpleNamespace

from imputegap.wrapper.AlgoPython.WGAN.models import WGAN_GRUI

warnings.simplefilter(action='ignore', category=FutureWarning)


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Enables TensorFlow 1.x behavior in TF2import argparse
import numpy as np
from data_loader import DataLoader
import os

tf.compat.v1.set_random_seed(0)
np.random.seed(0)
random.seed(0)


def main(input_matrix):
    # parse arguments

    args = SimpleNamespace(
        batch_size=1,
        gen_length=1,
        impute_iter=100,
        pretrain_epoch=5,
        g_loss_lambda=0.1,
        beta1=0.5,
        lr=0.001,
        epoch=30,
        n_inputs=1,
        n_hidden_units=1,
        n_classes=1,
        z_dim=1,
        isNormal=True,
        isBatch_normal=True,
        isSlicing=True,
        disc_iters=8,
        shape=None
    )

    row, col = input_matrix.shape
    args.shape = (row, col)
    args.n_inputs = col

    epochs=[args.epoch]
    g_loss_lambdas=[args.g_loss_lambda]
    beta1s = [0.5]

    for beta1 in beta1s:
        for e in epochs:
            for g_l in g_loss_lambdas:
                dt_train = DataLoader(input_matrix)  # Data loading
                tf.keras.backend.clear_session()  # Clears any previous models

                config = tf.compat.v1.ConfigProto()
                config.gpu_options.allow_growth = True

                with tf.compat.v1.Session(config=config) as sess:  # Create session
                    gan = WGAN_GRUI.WGAN(sess=sess, args=args, datasets=dt_train)  # Pass session

                    # build graph
                    print("Building model")
                    gan.build_model()

                    # launch the graph in a session
                    print("Starting training")
                    gan.train()
                    print(" [*] Training finished!")

                    print(" [*] Train dataset Imputation begin!")
                    imputed_matrix = gan.imputation(dt_train, True)
                    print(" [*] Train dataset Imputation finished!")


if __name__ == '__main__':

    ts_1 = TimeSeries()

    # 2. load the timeseries from file or from the code
    ts_1.load_series(utils.search_path("eeg-alcohol"))
    ts_1.normalize(normalizer="min_max")

    # 3. contamination of the data
    ts_mask = ts_1.Contamination.missing_percentage(ts_1.data)

    main(ts_mask)