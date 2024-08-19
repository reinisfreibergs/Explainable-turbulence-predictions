
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
sys.path.insert(0, '../conf')
sys.path.insert(0, '../models')


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

input_names = [r'$\tau_{wx}$', r'$\tau_{wz}$', r'$p_{w}$']
direction_names = ['$u$', '$v$', '$w$']
yp = 15

# data_input = pd.read_csv(r".\structure_data\structure_stats_abs_inputs_yp15_channel_0_uvw_0_outputs_ranked_by_shap_step_by_10.csv")
# data_filtered = data_input[data_input['area'] > 15]
#
# data_input_2 = pd.read_csv(r".\structure_data\structure_stats_abs_inputs_yp15_channel_0_uvw_1_outputs_ranked_by_shap_step_by_10.csv")
# data_filtered_2 = data_input_2[data_input_2['area'] > 15]


SAVE_MODE = False
area_limit = 13
for uvw in range(3):
    for channel in tqdm(range(3)):

        data_input_full = pd.read_csv(
            rf".\structure_data\structure_stats_abs_inputs_yp15_channel_{channel}_uvw_{uvw}_outputs_ranked_by_shap_step_by_10.csv")
        if SAVE_MODE:
            data = data_input_full[data_input_full['area'] > area_limit]
            data.to_csv(
                rf".\structure_data\structure_stats_abs_inputs_yp15_channel_{channel}_uvw_{uvw}_outputs_ranked_by_shap_step_by_10_area_13.csv",
                index=False)
            print('saved')
            continue

        # structure count plot
        # areas_of_interest = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        # plt.plot(areas_of_interest, [len(data_input_full[data_input_full['area'] > area_limit]) for area_limit in
        #           areas_of_interest])
        # plt.xlabel('Area limit')
        # plt.ylabel('Number of structures')


        data_input = pd.read_csv(
            rf".\structure_data\structure_stats_abs_inputs_yp15_channel_{channel}_uvw_{uvw}_outputs_ranked_by_shap_step_by_10_area_13.csv")

        data_filtered = data_input[data_input['area'] > area_limit]

        x = np.unique(data_filtered['step_start'])
        y_bin_full = np.bincount(np.array(data_input_full['step_start']*10, dtype=np.int32))
        y_bin = np.bincount(np.array(data_filtered['step_start']*10, dtype=np.int32))
        y = y_bin / np.sum(y_bin)
        y_diff = (y_bin_full - y_bin)/ np.sum(y_bin)
        plt.bar(x+0.05, y, width=0.095, alpha=0.9, label=f'{input_names[channel]}-{direction_names[uvw]}', color=f'C{channel}')

if not SAVE_MODE:
    plt.tight_layout()
    plt.ylabel(rf'Probability')
    plt.xlabel(rf'$\phi$ threshold')
    plt.xticks(np.arange(0, 1.1, 0.1))
    # plt.legend()
    plt.savefig(f'./bincount_{area_limit}.png', bbox_inches='tight')
    k = 0