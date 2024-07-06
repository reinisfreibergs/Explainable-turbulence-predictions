#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import time
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import platform
if platform.system() == 'Windows':
    import matplotlib
    matplotlib.use("qt5agg")

from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import shap
from tqdm import tqdm
from skimage.transform import resize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, '../conf')
sys.path.insert(0, '../models')

# Training utils
from src.training_utils import load_trained_model
from src.tfrecord_utils import get_dataset
# from conf.config_sample import WallRecon

import config


yp = 15


data_shap = np.load(rf'final_shap_{yp}.npy')
data_x = np.load(rf'./x_test_{yp}.npy')
data_y = np.load(rf'./y_test_{yp}.npy')
mean_shap = np.mean(np.abs(data_shap),axis=1)

input_names = [r'$\tau_{wx}$ modified input', r'$\tau_{wz}$ modified input', r'$p_{w}$ modified input']
direction_names = ['u', 'v', 'w']
USE_ALL_CHANNELS_FOR_REF = True
for index_uvw in range(3):
    relative_increases = [np.array([]), np.array([]), np.array([])]
    for sample_idx in tqdm(range(100)):
        result_row_init = np.load(f'./result_rows_yp_{yp}/result_row_uvw_{index_uvw}_sample_{sample_idx}.npz')
        result_row = [result_row_init[key] for key in result_row_init]


        index_u_v_w, first_np, second_total, mse_total, true_value, ranked_indices, sample_idx, pixel_frequency, idxes = result_row
        reference_mse_by_channel = np.mean((true_value - first_np.squeeze()) ** 2, axis=(-1, -2))
        if USE_ALL_CHANNELS_FOR_REF:
            reference_mse = np.mean(reference_mse_by_channel)
        else:
            reference_mse = reference_mse_by_channel[index_uvw]

        for channel_idx in range(3):
            if USE_ALL_CHANNELS_FOR_REF:
                modified_image_mse = np.mean(mse_total[channel_idx], axis=-1)
            else:
                modified_image_mse = mse_total[index_uvw, :, channel_idx]

            relative_increase = ((modified_image_mse - reference_mse) / reference_mse) * 100

            if int(sample_idx) == 0:
                relative_increases[channel_idx] = relative_increase
            else:
                relative_increases[channel_idx] = np.vstack((relative_increases[channel_idx], relative_increase))

        k = 0



    # plt.plot(idxes, relative_increase, label=input_names[channel_idx], linewidth=2)
    for channel_idx in range(3):
        plt.plot(idxes, np.mean(relative_increases[channel_idx], axis=0), label=input_names[channel_idx], linewidth=2)

    plt.legend(fontsize=24)
    # plt.title(rf'Change in MSE of the ${direction_names[index_u_v_w]}$ prediction after removing a fraction of pixels ordered by absolute SHAP', fontsize=24)
    plt.xlabel(rf'Fraction of pixels removed from input', fontsize=24)
    # plt.ylabel(rf'Relative increase in MSE, %', fontsize=24)
    plt.title(r'Relative increase in MSE, %: $\frac{{\text{mse_{orig}} - \text{mse_{removed}}}}{{\text{mse_{orig}}}} \times 100$', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    k = 0

