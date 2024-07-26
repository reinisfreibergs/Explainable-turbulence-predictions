#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import time

import pandas as pd
import sys
import matplotlib.pyplot as plt
import platform
if platform.system() == 'Windows':
    import matplotlib
    matplotlib.use("qt5agg")

import shap
from tqdm import tqdm
import cv2

# tf.config.run_functions_eagerly(True)

sys.path.insert(0, '../conf')
sys.path.insert(0, '../models')

# Training utils
from src.training_utils import load_trained_model
from src.tfrecord_utils import get_dataset
# from conf.config_sample import WallRecon
from sklearn.linear_model import LinearRegression
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# %% Configuration import
import config
# import scienceplots
# plt.style.use('science')

plt.rcParams['figure.figsize'] = (3.5, 2.5)
plt.rcParams['figure.dpi'] = 300
# Font settings
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# Axes and labels
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['lines.linewidth'] = 1.0
# Legends and titles
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.titlesize'] = 12
import scipy
from tqdm import tqdm
from scipy.stats import gaussian_kde

def visc_u(length_pixels, total_pixels=192, real_length=4 * np.pi, Re_tau=180):
    real_length_per_pixel = real_length / total_pixels
    length_real = length_pixels * real_length_per_pixel
    length_viscous_units = length_real * Re_tau
    return length_viscous_units

yp = 15
data_shap = np.load(rf'final_shap_{yp}.npy')
data_x = np.load(rf'./x_test_{yp}.npy')
data_y = np.load(rf'./y_test_{yp}.npy')

data_input = pd.read_csv(r".\structure_data\structure_stats_abs_inputs_yp15_channel_0_uvw_0_outputs_ranked_by_shap_step_by_10.csv")
data_filtered = data_input[data_input['area'] > 10]
column_names = data_filtered.columns.tolist()


step_thresh_max = data_filtered['step_start'] > 0.5
step_thresh_min = data_filtered['step_start'] < 0.5

np.array(data_filtered['shap_abs_sum'][step_thresh_max])
np.array(data_filtered['shap_abs_sum'][step_thresh_min])
np.array(data_filtered['length'])


x = np.unique(data_filtered['step_start'])
y_bin = np.bincount(np.array(data_filtered['step_start']*10, dtype=np.int32))
y = y_bin / np.sum(y_bin)
plt.fill_between(x, y, alpha=0.5)

data = data_filtered['area'].values
kde = gaussian_kde(data)
x_range = np.linspace(min(data), max(data), len(data))
pdf = kde(x_range)

# Normalize the PDF to ensure the integral is 1
pdf_normalized = pdf / np.sum(pdf * np.diff(x_range)[0])



plt.xlabel(rf'SHAP threshold')
plt.ylabel(rf'Structure count')
plt.tight_layout()
plt.show()



plt.scatter(visc_u(np.array(data_filtered['area'][step_thresh_min])), np.array(data_filtered['shap_abs_sum'][step_thresh_min]), s=1, color='blue', label='Lowest 50% by rank')
plt.scatter(visc_u(np.array(data_filtered['area'][step_thresh_max])) ,np.array(data_filtered['shap_abs_sum'][step_thresh_max]), s=1, color='red', label='Highest 50% by rank')
plt.xlabel(r'$\Delta x$, $L^{+}$')
plt.ylabel(r'$|\phi|$')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.scatter(np.array(data_filtered['shap_abs_sum'][step_thresh_min]), visc_u(np.array(data_filtered['area'][step_thresh_min])), s=1, color='blue', label='Lowest 50% by rank')
plt.scatter(np.array(data_filtered['shap_abs_sum'][step_thresh_max]), visc_u(np.array(data_filtered['area'][step_thresh_max])), s=1, color='red', label='Highest 50% by rank')
plt.ylabel(r'$\Delta x$, $L^{+}$')
plt.xlabel(r'$log |\phi|$')
plt.xscale('log')
plt.legend()
plt.tight_layout()


X = visc_u(np.array(data_filtered['area']))
Y = np.array(data_filtered['shap_abs_sum'])/X
cmap = plt.get_cmap('tab10')  # 'tab10' colormap has 10 distinct colors
unique_X = np.unique(data_filtered['step_start'])
color_map = {val: cmap(i) for i, val in enumerate(unique_X)}


# for val in unique_X:
#     plt.scatter(X[data_filtered['step_start']==val], Y[data_filtered['step_start']==val], color=color_map[val], label=f'X = {val}', s=1)
# plt.yscale('log')

for val in unique_X:
    # plt.scatter(X[data_filtered['step_start'] == val], Y[data_filtered['step_start'] == val], color=color_map[val], label=f'X = {val}', s=1)
    X_p = X[data_filtered['step_start'] == val]
    Y_p = Y[data_filtered['step_start'] == val]
    hist, xedges, yedges = np.histogram2d(X_p, Y_p, bins=[100, 100])
    # plt.imshow(hist.T / np.sum(hist.T) > 0.0001)
    x_hist, y_hist = np.where(hist.T / np.sum(hist.T) > 0.0001)
    plt.scatter(xedges[x_hist], yedges[y_hist], color=color_map[val], label=f'X = {val}', s=1)
# plt.yscale('log')
plt.legend()


def create_contour_plot(X, Y):
    xmin, xmax = 0.99*X.min(), 1.01*X.max()
    ymin, ymax = 0.99*Y.min(), 1.01*Y.max()
    bins = (100, 100)  # Number of bins for the histogram
    H_init, xedges, yedges = np.histogram2d(X, Y, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
    H = scipy.ndimage.gaussian_filter(H_init, sigma=2)

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    X_grid, Y_grid = np.meshgrid(xcenters, ycenters)

    num_levels = 10
    min_density = H[H > 0].min()  # Minimum non-zero density
    max_density = H.max()  # Maximum density
    levels = np.linspace(min_density, max_density, num_levels)

    # Plot the 2D histogram
    plt.figure(figsize=(22, 10))
    plt.imshow(H.T, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='viridis', aspect='auto')
    # plt.colorbar(label='Count')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.title('2D Histogram Density Plot')
    # plt.scatter(X, Y)
    contour = plt.contour(X_grid, Y_grid, H.T, levels=levels, colors='red', linewidths=0.2, extent=[xmin, xmax, ymin, ymax])
    contourf = plt.contourf(X_grid, Y_grid, H.T, levels=levels, extent=[xmin, xmax, ymin, ymax])


    k = 0
    # plt.savefig(rf'./images/area_shap_abs_sum_yp15_channel_{channel}_uvw_{uvw}.png')
    plt.close()



k = 0