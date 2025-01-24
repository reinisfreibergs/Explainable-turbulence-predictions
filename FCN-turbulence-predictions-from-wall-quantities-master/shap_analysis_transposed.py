import numpy as np
import os
import matplotlib.pyplot as plt
import platform
if platform.system() == 'Windows':
    import matplotlib
    matplotlib.use("qt5agg")
from matplotlib.ticker import ScalarFormatter
from skimage.transform import resize
from mpl_toolkits.axes_grid1 import ImageGrid
plt.rcParams['figure.figsize'] = (2.5, 3.5)
plt.rcParams['figure.dpi'] = 150
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

plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
# turn on latex
plt.rc('text', usetex=True)


# copied from main_v3 - but this is for the outputs
default_scale = np.array([[ 1.00000000e+00,  4.65725183e-01],
 [ 1.69644458e-01,  5.77261763e-21],
 [ 3.63199450e-01, -1.10760695e-04]])

# for inputs one must use ones taken from tfrecord_utils
# ds_path = app.DS_PATH_TEST
ds_path = './storage/Test'
avg_input_path = ds_path + '/.avg_inputs/'
print('The inputs are normalized to have a unit Gaussian distribution')
# just taken the values from app directly, without reloading for the 100th time
N_DATASETS = 2
INTERV_TRAIN = 3
with np.load(avg_input_path +
             f'stats_ds{N_DATASETS}x4200' +
             f'_dt{int(0.45 * 100 * INTERV_TRAIN)}.npz') as data:
    avgs_in = data['mean_inputs'].astype(np.float32)
    std_in = data['std_inputs'].astype(np.float32)

k = 0

NORMALIZE_IN = True
NORMALIZE_SHAP = True

for yp in [15, 100]:
    data_shap = np.load(rf'final_shap_{yp}.npy')
    data_x = np.load(rf'./x_test_{yp}.npy')
    data_y = np.load(rf'./y_test_{yp}.npy')


    # for amount_of_samples in [1, 10, 100, 1000]:
    for amount_of_samples in [1]:
        # mean over the 1000 samples - (3, 3, 208, 208)
        # remove the 15pix padding
        # this works even with a single sample as then it only squeezes the dimension
        mean_shap = np.mean(np.abs(data_shap[:, :amount_of_samples, :, 8:-8, 8:-8]), axis=1)
        # mean_shap = np.mean((data_shap[:, :amount_of_samples, :, 8:-8, 8:-8]), axis=1)
        mean_x = (np.mean(data_x[:amount_of_samples, :, 8:-8, 8:-8], axis=0))

        if NORMALIZE_IN:
            mean_x = (mean_x - np.mean(mean_x, axis=(-1, -2))[:, None, None]) / np.std(mean_x, axis=(-1, -2))[:, None, None]
        if NORMALIZE_SHAP:
            # shap_mean_value_over_channels = np.sqrt(np.sum(mean_shap ** 2, axis=(0))) / 192 ** 2
            # mean_shap = (mean_shap) / shap_mean_value_over_channels[None, :, :, :]
            # shap_mean_value_over_channels = np.sqrt(np.sum(mean_shap**2, axis=(-1, -2, -3))) / 192**2
            shap_mean_value_over_channels = np.mean(mean_shap, axis=(-1, -2, -3))
            mean_shap = (mean_shap) / shap_mean_value_over_channels[:, None, None, None]

        input_names = [r'$\tau_{wx}$', r'$\tau_{wz}$', r'$p_{w}$']
        direction_names = ['$u$', '$v$', '$w$']

        # Create a figure and axes
        fig, axes = plt.subplots(nrows=4, ncols=3)  # Adjust figsize as needed
        for j in range(3):
            # added after 18.24.2024 request for normalization in 1 mean, 0 std
            final_mean, final_std = default_scale[j][1], default_scale[j][0]
            for i in range(4):
                ax = axes[i, j]

                # domain size is (4pi*h, h, 2pi*h)
                if i == 0:
                    data = resize(mean_x[j], (192, 384))

                    if NORMALIZE_IN:
                        data_std_limit = 1.5
                        data = np.clip(data, -data_std_limit, data_std_limit)

                    body = f'{input_names[j]}'
                    ax.set_title(fr'Input $\overline{{{input_names[j][1:-1]}}}$', bbox=dict(pad=0, facecolor='none', edgecolor='none'))
                    if NORMALIZE_IN:
                        ax.set_title(fr'Input $\widetilde{{{input_names[j][1:-1]}}}$',bbox=dict(pad=0, facecolor='none', edgecolor='none'))
                    # ax.set_ylabel(r'$z/h$')
                    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False, labelsize=10)
                    ax.tick_params(axis='y', which='both', left=False, top=False, labelleft=False, labelbottom=False, labeltop=False, labelsize=10)
                    # ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    # ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                    # cmap = 'RdBu'
                    cmap = 'viridis'
                else:
                    data = resize(mean_shap[i-1][j], (192, 384))
                    ax.set_title(fr'$\overline{{|\phi_{{{direction_names[i-1][1:-1]}, {input_names[j][1:-1]}}}|}}$', bbox=dict(pad=0, facecolor='none', edgecolor='none'))
                    if NORMALIZE_SHAP:
                        ax.set_title(
                            fr'${{|\widehat{{\phi}}_{{{direction_names[i-1][1:-1]}, {input_names[j][1:-1]}}}|}}$',
                            bbox=dict(pad=0, facecolor='none', edgecolor='none'))
                    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False, labelsize=10)
                    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False, labelsize=10)
                    cmap = 'binary'

                if i == 3:
                    ax.set_xlabel(r'$x/h$')
                    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False, labelsize=10)
                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                if j == 0:
                    ax.set_ylabel(r'$z/h$')
                    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False, labelsize=10)
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

                im = ax.imshow(data, cmap=cmap, origin='lower', extent=[-2*np.pi, 2*np.pi, -np.pi, np.pi])
                cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.2, pad=0.04, aspect=30)
                cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))

                if i == 0:
                    cbar.set_ticks([-1.5, 1.5])  # Set tick positions
                    cbar.set_ticklabels([r'\textless{}-1.5', r'\textgreater{}1.5'])
                else:
                    cbar.set_ticks([np.min(data), np.max(data)])  # Set ticks for the colorbar

                if i != 0:
                    cbar.ax.set_yticklabels(['{:.1e}'.format(np.min(data)), '{:.1e}'.format(np.max(data))])  # Set tick labels
                cbar.ax.tick_params(axis='y', labelsize=10)
                # cbar ax set font sizze

        # wspace 0.07 or top 0.975 and wspace 0.0
        # plt.tight_layout()
        plt.subplots_adjust(wspace=0.07, top=0.975)
        plt.subplots_adjust(wspace=0.0)
        # plt.savefig(f'./images/shap_analysis_{yp}_over_{amount_of_samples}_samples.png', bbox_inches='tight', dpi=300)
        # plt.savefig(f'./mean_shap_analysis_{yp}_over_{amount_of_samples}_samples_v2.png', bbox_inches='tight', dpi=300)
        # plt.savefig(f'./mean_shap_analysis_{yp}_over_{amount_of_samples}_samples_v3_single_all_normalization_v2.png', bbox_inches='tight', dpi=300)
        # plt.show()

        # cmap='coolwarm'
        k = 0