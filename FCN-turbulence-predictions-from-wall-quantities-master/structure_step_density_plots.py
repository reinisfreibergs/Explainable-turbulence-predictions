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
from matplotlib.lines import Line2D
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
# add latex
plt.rc('text', usetex=True)
import scipy
from tqdm import tqdm
from scipy.stats import gaussian_kde


u_rms = np.load(r'./u_rms.npy')
default_scale = np.load(r'./default_scale.npy')
u_v_w = 0
final_mean, final_std = default_scale[u_v_w][1], default_scale[u_v_w][0]

def default_scale_predicted_output(pred_init):
    # # Revert back to the flow field
    pred = np.array(pred_init)
    for i in range(3):
        final_mean, final_std = default_scale[i][1], default_scale[i][0]
        pred[:, i] *= final_std
    return pred
def create_contour_plot(X, Y, min_h = 0.01, cmap='viridis', return_img=False, sigma=1):
    # X = (X - np.mean(X)) / np.std(X)
    input_names = [r'$\tau_{wx}$', r'$\tau_{wz}$', r'$p_{w}$']
    direction_names = ['$u$', '$v$', '$w$']
    # Create grid points for KDE
    xmin, xmax = 0.99 * X.min(), 1.01 * X.max()
    ymin, ymax = 0.99 * Y.min(), 1.01 * Y.max()
    bins = (200, 200)  # Number of bins for the histogram
    H_init_r, xedges, yedges = np.histogram2d(X, Y, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
    # divide by sum
    H_init = H_init_r / np.sum(H_init_r)
    # normalize to [0,1] interval
    # H_init = H_init / np.max(H_init)
    # H_init = H_init * (H_init > 0.1)
    H_inf = np.array(H_init)
    H_inf[H_inf < 5e-5] = -np.inf
    # H_init[H_init < 1e-4] = -np.inf

    H = scipy.ndimage.gaussian_filter(H_init, sigma=sigma)
    H[H < 5e-5] = -np.inf
    # H = H * (H > min_h)

    if not return_img:
        # plt.figure()
        im = plt.imshow(np.log10(H.T), origin='lower', extent=[xmin, xmax, ymin, ymax], cmap=cmap, aspect='auto')
    else:
        return H.T, xmin, xmax, ymin, ymax
    k = 0
def visc_u(length_pixels, total_pixels=192, real_length=4 * np.pi, Re_tau=180):
    real_length_per_pixel = real_length / total_pixels
    length_real = length_pixels * real_length_per_pixel
    length_viscous_units = length_real * Re_tau
    return length_viscous_units

def visc_u_area(area_pixels, total_pixels=192, Re_tau=180):
    real_length_per_pixel_x = 4 * np.pi / total_pixels
    real_length_per_pixel_y = 2 * np.pi / total_pixels
    length_viscous_units_x = real_length_per_pixel_x * Re_tau
    length_viscous_units_y = real_length_per_pixel_y * Re_tau
    viscous_area = area_pixels * (length_viscous_units_x * length_viscous_units_y)

    return viscous_area

def find_valid_bounds(image):
    # Find the indices of valid (non-NaN) values
    valid_indices = np.argwhere(~np.isinf(image))

    if valid_indices.size == 0:
        return None, None, None, None

    # Extract x and y coordinates
    y_indices, x_indices = valid_indices[:, 0], valid_indices[:, 1]

    # Determine the minimum and maximum x and y coordinates
    xmin = np.min(x_indices)
    xmax = np.max(x_indices)
    ymin = np.min(y_indices)
    ymax = np.max(y_indices)

    return xmin, xmax, ymin, ymax


def zoom_in_image(image, extent, zoom_pixels):
    image_height, image_width = image.shape
    xmin, xmax, ymin, ymax = extent
    zoom_xmin_pixel, zoom_xmax_pixel, zoom_ymin_pixel, zoom_ymax_pixel = zoom_pixels

    x_ratio = (xmax - xmin) / image_width
    y_ratio = (ymax - ymin) / image_height

    zoom_xmin_data = xmin + zoom_xmin_pixel * x_ratio
    zoom_xmax_data = xmin + zoom_xmax_pixel * x_ratio
    zoom_ymin_data = ymin + (image_height - zoom_ymax_pixel) * y_ratio
    zoom_ymax_data = ymin + (image_height - zoom_ymin_pixel) * y_ratio

    return zoom_xmin_data, zoom_xmax_data, zoom_ymin_data, zoom_ymax_data

def create_plot(input1x, input2x, input1y, input2y, x_name, sigma=1, savename='j'):
    colors = ['#440154', '#31688e', '#a0da39']
    plt.figure(figsize=(3.5, 2.5))
    image1, xmin1, xmax1, ymin1, ymax1 = create_contour_plot(input1x, input1y, min_h=0.05, return_img=True, sigma=sigma)
    image2, xmin2, xmax2, ymin2, ymax2 = create_contour_plot(input2x, input2y, min_h=0.05, return_img=True, sigma=sigma)
    # recalculate the min and max values from image1 and image2
    xmax, xmin, ymax, ymin = max(xmax1, xmax2), min(xmin1, xmin2), max(ymax1, ymax2), min(ymin1, ymin2)


    # img = plt.imshow(np.zeros_like(image1), cmap='binary', alpha=0.0, origin='lower', label='1', extent=[xmin, xmax, ymin, ymax], aspect='auto')
    # plt.imshow(image1, cmap='viridis', alpha=1.0, origin='lower', label='1', extent=[xmin, xmax, ymin, ymax], aspect='auto')
    # plt.imshow(image2, cmap='cividis', alpha=1.0, origin='lower', label='2', extent=[xmin, xmax, ymin, ymax], aspect='auto')
    plt.imshow(image1, cmap='Reds', alpha=1.0, origin='lower', label='1', extent=[xmin, xmax, ymin, ymax], aspect='auto')
    plt.imshow(image2, cmap='Blues', alpha=1.0, origin='lower', label='2', extent=[xmin, xmax, ymin, ymax], aspect='auto')
    # plt.legend()

    def find_maxlim(no=1):
        # map this to the original data xmax and ymax
        if no == 1:
            maxylim, maxxlim = [i.max() for i in np.where(1 - 1 * np.isinf(image1))]
            maxxlim = xmin1 + maxxlim * (xmax1 - xmin1) / image1.shape[1]
            maxylim = ymin1 + maxylim * (ymax1 - ymin1) / image1.shape[0]
        else:
            maxylim, maxxlim = [i.max() for i in np.where(1 - 1 * np.isinf(image2))]
            maxxlim = xmin2 + maxxlim * (xmax2 - xmin2) / image2.shape[1]
            maxylim = ymin2 + maxylim * (ymax2 - ymin2) / image2.shape[0]

        return maxxlim, maxylim

    def find_minlim(no=1):
        # map this to the original data xmax and ymax
        if no==1:
            maxylim, maxxlim = [i.min() for i in np.where(1 - 1 * np.isinf(image1))]
            maxxlim = xmin1 + maxxlim * (xmax1 - xmin1) / image1.shape[1]
            maxylim = ymin1 + maxylim * (ymax1 - ymin1) / image1.shape[0]
        else:
            maxylim, maxxlim = [i.min() for i in np.where(1 - 1 * np.isinf(image2))]
            maxxlim = xmin2 + maxxlim * (xmax2 - xmin2) / image2.shape[1]
            maxylim = ymin2 + maxylim * (ymax2 - ymin2) / image2.shape[0]

        return maxxlim, maxylim

    def find_maxlim_combined(image1, image2):
        maxylim1, maxxlim1 = find_maxlim(1)
        maxylim2, maxxlim2 = find_maxlim(2)
        maxxlim = max(maxxlim1, maxxlim2)
        maxylim = max(maxylim1, maxylim2)
        return maxxlim, maxylim

    def find_minlim_combined(image1, image2):
        maxylim1, maxxlim1 = find_minlim(1)
        maxylim2, maxxlim2 = find_minlim(2)
        maxxlim = min(maxxlim1, maxxlim2)
        maxylim = min(maxylim1, maxylim2)
        return maxxlim, maxylim

    maxylim, maxxlim = find_maxlim_combined(image1, image2)
    minylim, minxlim = find_minlim_combined(image1, image2)


    plt.xlim([minxlim, maxxlim])
    plt.ylim([minylim, maxylim])
    plt.xlabel(x_name)
    # plt.ylabel(r'$\frac{|\Phi|}{A}$')
    plt.ylabel(rf"$\phi_{{{direction_names[u_v_w]}, {input_names[channel]}}}$ $/ A$")
    norm = plt.Normalize(vmin=np.min(image2[np.where(1-1*np.isinf(image2))]), vmax=np.max(image1))
    axs = plt.gca()
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='Greys'), ax=axs)
    # cb1.formatter.set_powerlimits((0, 0))
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    def log_format(x, pos):
        if x == 0:
            return '0'
        exponent = int(np.floor(np.log10(abs(x))))
        mantissa = x / 10 ** exponent
        return r'${:.0f}\cdot10^{{{}}}$'.format(mantissa, exponent)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(log_format))
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.set_title('frequency', fontsize=8)


    # legend_elements = [Line2D([0], [0], color=colors[0], lw=4, label='Upper deciles'),
    #                    Line2D([0], [0], color=colors[1], lw=4, label='Lower deciles')]
    legend_elements = [Line2D([0], [0], color='red', lw=4, label='Upper deciles'),
                       Line2D([0], [0], color='blue', lw=4, label='Lower deciles')]
    plt.gca().legend(handles=legend_elements, fontsize=6.5)


    plt.tight_layout()
    # plt.show()
    plt.savefig(rf'structure_step_10_density_yp15_area_{area_limit}_channel_{channel}_uvw_{u_v_w}_{savename}.png', dpi=300, bbox_inches='tight')
    plt.close()

yp = 15
data_shap = np.load(rf'final_shap_{yp}.npy')
data_x = np.load(rf'./x_test_{yp}.npy')
data_y = np.load(rf'./y_test_{yp}.npy')

# channel = 2
# u_v_w = 0
area_limit = 13
for u_v_w in range(3):
    for channel in tqdm(range(3)):
        # data_input = pd.read_csv(rf".\structure_data\structure_stats_abs_inputs_yp15_channel_{channel}_uvw_{u_v_w}_outputs_ranked_by_shap_step_by_10_area_{area_limit}.csv")
        data_input = pd.read_csv(
            rf".\structure_data\structure_stats_abs_inputs_yp15_channel_{channel}_uvw_{uvw}_outputs_ranked_by_shap_step_by_10.csv")
        data_filtered = data_input[data_input['area'] > area_limit]
        column_names = data_filtered.columns.tolist()


        step_thresh_max = data_filtered['step_start'] > 0.5
        step_thresh_min = data_filtered['step_start'] < 0.5

        shap_abs_max = np.array(data_filtered['shap_abs_sum'][step_thresh_max])
        shap_abs_min = np.array(data_filtered['shap_abs_sum'][step_thresh_min])
        area_max = np.array(data_filtered['area'][step_thresh_max])
        area_min = np.array(data_filtered['area'][step_thresh_min])

        length_max = visc_u(np.array(data_filtered['length'][step_thresh_max]))
        length_min = visc_u(np.array(data_filtered['length'][step_thresh_min]))
        heigth_max = visc_u(np.array(data_filtered['height'][step_thresh_max]))
        heigth_min = visc_u(np.array(data_filtered['height'][step_thresh_min]))

        original_abs_max = np.array(data_filtered['original_abs_sum'][step_thresh_max])
        original_abs_min = np.array(data_filtered['original_abs_sum'][step_thresh_min])

        original_min = np.array(data_filtered['original_sum'][step_thresh_min])
        original_max = np.array(data_filtered['original_sum'][step_thresh_max])

        output_abs_max = np.array(data_filtered['output_abs_sum'][step_thresh_max])
        output_abs_min = np.array(data_filtered['output_abs_sum'][step_thresh_min])

        output_max = np.array(data_filtered['output_sum'][step_thresh_max])
        output_min = np.array(data_filtered['output_sum'][step_thresh_min])

        # create_contour_plot(original_abs_max/area_max, shap_abs_max/area_max, min_h=0.01)
        # create_contour_plot(original_abs_min/area_min, shap_abs_min/area_min, min_h=0.01, cmap='ciridis')

        input_names = [r'\tau_{wx}', r'\tau_{wz}', r'p_{w}']
        direction_names = ['u', 'v', 'w']


        wall_shear_stress = (180 / (2*2100))**2

        create_plot(original_abs_max/area_max, original_abs_min/area_min, shap_abs_max/area_max, shap_abs_min/area_min, rf'$\frac{{|{input_names[channel]}|}}{{A}}$', savename='orig_abs')
        create_plot(length_max, length_min, shap_abs_max/area_max, shap_abs_min/area_min, r'$\Delta x$, $L^{+}$', sigma=3, savename='length')
        create_plot(heigth_max, heigth_min, shap_abs_max/area_max, shap_abs_min/area_min, r'$\Delta y$, $L^{+}$', sigma=3, savename='height')

        create_plot(area_max, area_min, shap_abs_max/area_max, shap_abs_min/area_min, r'$A$', savename='just_area')
        # plot with aspect ratio
        create_plot(length_max/heigth_max, length_min/heigth_min, shap_abs_max/area_max, shap_abs_min/area_min, r'$\Delta y / \Delta x$, $L^{+}$', sigma=3, savename='aspect ratio')


        # create_plot(original_max/(area_max*final_std*wall_shear_stress), original_min/(area_min*final_std*wall_shear_stress), shap_abs_max/area_max, shap_abs_min/area_min, rf'${input_names[channel]}/A \ \frac{1}{\tau_{\text{wall}}}$', savename='orig_wall')
        create_plot(original_max/area_max, original_min/area_min, shap_abs_max/area_max, shap_abs_min/area_min, rf'$\frac{{{input_names[channel]}}}{{A}}$', savename='orig')
        create_plot(output_abs_max/area_max, output_abs_min/area_min, shap_abs_max/area_max, shap_abs_min/area_min, rf'$\frac{{|{direction_names[u_v_w]}|}}{{A}}$', savename='output_abs')
        create_plot(output_max/area_max, output_min/area_min, shap_abs_max/area_max, shap_abs_min/area_min, rf'$\frac{{|{direction_names[u_v_w]}|}}{{A}}$', savename='output_abs')

k = 0



# plt.scatter(visc_u(np.array(data_filtered['area'][step_thresh_min])), np.array(data_filtered['shap_abs_sum'][step_thresh_min]), s=1, color='blue', label='Lowest 50% by rank')
# plt.scatter(visc_u(np.array(data_filtered['area'][step_thresh_max])) ,np.array(data_filtered['shap_abs_sum'][step_thresh_max]), s=1, color='red', label='Highest 50% by rank')
# plt.xlabel(r'$\Delta x$, $L^{+}$')
# plt.ylabel(r'$|\phi|$')
# plt.legend()
# plt.tight_layout()
# plt.show()
#
# plt.figure()
# plt.scatter(np.array(data_filtered['shap_abs_sum'][step_thresh_min]), visc_u(np.array(data_filtered['area'][step_thresh_min])), s=1, color='blue', label='Lowest 50% by rank')
# plt.scatter(np.array(data_filtered['shap_abs_sum'][step_thresh_max]), visc_u(np.array(data_filtered['area'][step_thresh_max])), s=1, color='red', label='Highest 50% by rank')
# plt.ylabel(r'$\Delta x$, $L^{+}$')
# plt.xlabel(r'$log |\phi|$')
# plt.xscale('log')
# plt.legend()
# plt.tight_layout()
#
#
# plt.figure()
# plt.scatter(original_abs_min/area_min, shap_abs_min/area_min, s=1, color='blue', label='Lowest 50% by rank')
# plt.scatter(original_abs_max/area_max, shap_abs_max/area_max, s=1, color='red', label='Highest 50% by rank')
# plt.ylabel(r'shap/area')
# plt.xlabel(r'original/area')
# # plt.xscale('log')
# plt.legend()
# plt.tight_layout()
#
#
# X = visc_u(np.array(data_filtered['area']))
# Y = np.array(data_filtered['shap_abs_sum'])/X
# cmap = plt.get_cmap('tab10')  # 'tab10' colormap has 10 distinct colors
# unique_X = np.unique(data_filtered['step_start'])
# color_map = {val: cmap(i) for i, val in enumerate(unique_X)}
#
#
#
#
#
#
# k = 0