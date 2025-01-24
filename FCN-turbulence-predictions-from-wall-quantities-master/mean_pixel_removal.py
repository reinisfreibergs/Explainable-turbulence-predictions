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
plt.rcParams['figure.figsize'] = (3.5, 2.5)
plt.rcParams['figure.dpi'] = 300
# Font settings
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# Axes and labels
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
# Legends and titles
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rc('text', usetex=True)

yp = 15


data_shap = np.load(rf'final_shap_{yp}.npy')
data_x = np.load(rf'./x_test_{yp}.npy')
data_y = np.load(rf'./y_test_{yp}.npy')
mean_shap = np.mean(np.abs(data_shap),axis=1)

input_names = [r'$\tau_{wx}$', r'$\tau_{wz}$', r'$p_{w}$']
# input_names = [r'\tau_{wx}', r'\tau_{wz}', r'p_{w}']
# direction_names = ['$u$', '$v$', '$w$']
direction_names = ['u', 'v', 'w']
# colors = ['red', 'blue', 'green']
line_types = ['-', '--', ':']


plt.rc('text', usetex=True)
plt.rcParams['font.size'] = 10 + 2
plt.rcParams['font.family'] = 'serif'
# axs.plot(top_x, top_y, color=colors[uvw], linestyle=line_types[channel]
colors = ['#a0da39', '#440154', '#31688e']
USE_ALL_CHANNELS_FOR_REF = True
for index_uvw in range(3):
    relative_increases = [np.array([]), np.array([]), np.array([])]
    for sample_idx in tqdm(range(100)):
        result_row_init = np.load(f'./result_rows_yp_{yp}/result_row_uvw_{index_uvw}_sample_{sample_idx}.npz')
        result_row = [result_row_init[key] for key in result_row_init]


        index_u_v_w, first_np, second_total, mse_total, true_value, ranked_indices, sample_idx, pixel_frequency, idxes = result_row
        reference_mse_by_channel = np.mean((true_value - first_np.squeeze()) ** 2, axis=(-1, -2))
        original_max_pixel_value = np.max(first_np.squeeze(), axis=(-1, -2))
        mse_values = np.mean((true_value - second_total) ** 2, axis=(2, 3))
        if USE_ALL_CHANNELS_FOR_REF:
            reference_mse = np.mean(reference_mse_by_channel)
        else:
            reference_mse = reference_mse_by_channel[index_uvw]

        for channel_idx in range(3):
            if USE_ALL_CHANNELS_FOR_REF:
                # modified_image_mse = np.mean(mse_total[channel_idx], axis=-1)
                modified_image_mse = np.mean(mse_total[channel_idx] / original_max_pixel_value, axis=-1)
            else:
                # modified_image_mse = mse_total[index_uvw, :, channel_idx] / original_max_pixel_value[index_uvw]
                # modified_image_mse = mse_total[channel_idx, :, index_u_v_w] / original_max_pixel_value[index_u_v_w]
                modified_image_mse = mse_total[channel_idx, :, index_u_v_w]

            # relative_increase = ((modified_image_mse - reference_mse) / reference_mse) * 100
            # if np.mean(modified_image_mse) > 0.2:
            #     print('warning')
            #     modified_image_mse /= 10
            relative_increase = ((modified_image_mse))

            if int(sample_idx) == 0:
                relative_increases[channel_idx] = relative_increase
            else:
                relative_increases[channel_idx] = np.vstack((relative_increases[channel_idx], relative_increase))

        k = 0



        # plt.plot(idxes, relative_increase, label=input_names[channel_idx], linewidth=2)
    for channel_idx in range(3):
        plt.plot(idxes, (np.mean(relative_increases[channel_idx], axis=0)),
                 # label=f"$\phi({direction_names[index_u_v_w]}, {input_names[channel_idx]})$",
                 label=input_names[channel_idx],
                 linewidth=1,
                 color=colors[channel_idx])
                 # linestyle=line_types[channel_idx])
    plt.legend(loc='best', fontsize=12)

# plt.title(rf'Change in MSE of the ${direction_names[index_u_v_w]}$ prediction after removing a fraction of pixels ordered by absolute SHAP', fontsize=24)
# plt.xlabel(rf'Fraction of pixels removed from input')
plt.xlabel(r'$\frac{{\rm{px}}_{\rm{rmv}}}{{\rm{px}}_{\rm{tot}}}$', fontsize=14)
plt.ylabel(r'$\rm{MSE}^*$', fontsize=12)
# plt.title(r'Relative increase in MSE, %: $\frac{{\text{mse_{orig}} - \text{mse_{removed}}}}{{\text{mse_{orig}}}} \times 100$', fontsize=24)
# plt.title(r'MSE of velocity fluctuations with a modified input channel', fontsize=10)
plt.xticks()
plt.yticks(np.arange(0, 0.15, 0.015))
plt.tight_layout()
# plt.savefig(f'./relative_increase_mse_{yp}_{index_uvw}.png', bbox_inches='tight', dpi=300)
k = 0

