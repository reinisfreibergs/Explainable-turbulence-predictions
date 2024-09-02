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
plt.rcParams['figure.figsize'] = (3.5*2.5, 2.5*1.5)
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

plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
# turn on latex
plt.rc('text', usetex=True)

for yp in [15, 100]:
    data_shap = np.load(rf'final_shap_{yp}.npy')
    data_x = np.load(rf'./x_test_{yp}.npy')
    data_y = np.load(rf'./y_test_{yp}.npy')


    # for amount_of_samples in [1, 10, 100, 1000]:
    for amount_of_samples in [1000]:
        # mean over the 1000 samples - (3, 3, 208, 208)
        # remove the 15pix padding
        mean_shap = np.mean(np.abs(data_shap[:, :amount_of_samples, :, 8:-8, 8:-8]), axis=1)
        mean_x = np.mean(data_x[:amount_of_samples, :, 8:-8, 8:-8], axis=0)


        input_names = [r'$\tau_{wx}$', r'$\tau_{wz}$', r'$p_{w}$']
        direction_names = ['$u$', '$v$', '$w$']

        # Create a figure and axes
        fig, axes = plt.subplots(nrows=3, ncols=4)  # Adjust figsize as needed
        for i in range(3):
            for j in range(4):
                ax = axes[i, j]

                # domain size is (4pi*h, h, 2pi*h)
                if j == 0:
                    data = resize(mean_x[i], (192, 384))
                    body = f'{input_names[i]}'
                    ax.set_title(fr'Input $\overline{{{input_names[i][1:-1]}}}$', bbox=dict(pad=0, facecolor='none', edgecolor='none'))
                    ax.set_ylabel(r'$z/h$')
                    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False, labelsize=10)
                    ax.tick_params(axis='y', which='both', left=True, right=True, labelleft=True, labelright=False, labelsize=10)
                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                    cmap = 'RdBu'
                else:
                    data = resize(mean_shap[j-1][i], (192, 384))
                    ax.set_title(fr'$\overline{{|\phi_{{{direction_names[j-1][1:-1]}, {input_names[i][1:-1]}}}|}}$', bbox=dict(pad=0, facecolor='none', edgecolor='none'))
                    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False, labelsize=10)
                    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False, labelsize=10)
                    cmap = 'binary'

                if i == 2:
                    ax.set_xlabel(r'$x/h$')
                    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, labeltop=False, labelsize=10)
                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))

                im = ax.imshow(data, cmap=cmap, origin='lower', extent=[-2*np.pi, 2*np.pi, -np.pi, np.pi])
                cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
                cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
                cbar.set_ticks([np.min(data), np.max(data)])  # Set ticks for the colorbar
                cbar.ax.set_yticklabels(['{:.1e}'.format(np.min(data)), '{:.1e}'.format(np.max(data))])  # Set tick labels
                cbar.ax.tick_params(axis='y', labelsize=10)
                # cbar ax set font sizze

        plt.tight_layout()
        # plt.savefig(f'./images/shap_analysis_{yp}_over_{amount_of_samples}_samples.png', bbox_inches='tight', dpi=300)
        # plt.savefig(f'./mean_shap_analysis_{yp}_over_{amount_of_samples}_samples_v2.png', bbox_inches='tight', dpi=300)
        plt.show()

        # cmap='coolwarm'
        k = 0