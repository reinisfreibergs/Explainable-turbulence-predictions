import numpy as np
from scipy.ndimage import label
from tqdm import tqdm  # pip install tqdm
import pandas as pd
import matplotlib.pyplot as plt
import platform
if platform.system() == 'Windows':
    import matplotlib
    matplotlib.use("qt5agg")
from mpl_toolkits.axes_grid1 import ImageGrid

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
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.titlesize'] = 12
# add latex
plt.rc('text', usetex=True)

yp = 15
data_shap = np.load(rf'final_shap_{yp}.npy')
data_x = np.load(rf'./x_test_{yp}.npy')
data_y = np.load(rf'./y_test_{yp}.npy')




for u_v_w in range(1):
    for channel in range(2, 3):
        result_dict = {'area': [], 'length': [], 'height': [], 'original_abs_sum': [], 'original_sum': [],
                       'shap_abs_sum': [], 'shap_sum': [], 'sample_idx': [], 'channel': [], 'uvw': [],
                       'output_abs_sum': [], 'output_sum': [],
                       'mean_minimum_dist_to_wall': [], 'step_start': []}

        for sample_idx in [0]:
        # for sample_idx in tqdm(range(1)):

            sample = data_x[sample_idx][channel][8:-8, 8:-8]
            sample_output = data_y[sample_idx][u_v_w]
            sample_shap = data_shap[u_v_w][sample_idx][channel][8:-8, 8:-8]
            ranked_indices = np.array(np.unravel_index(np.argsort(np.abs(sample_shap).flatten()), sample_shap.shape)).T

            new_img = np.zeros_like(sample)
            new_img_shap = np.zeros_like(sample)

            total_structure_mask = np.zeros_like(sample)

            n = 10
            points = np.linspace(0, 1, n + 1)
            # Create the chunks by pairing consecutive points
            chunks = [[points[i], points[i + 1]] for i in range(n)]
            #
            # pick only the extreme ones, at least for now
            # chunks = [chunks[0]] + [chunks[-1]]
            fig, axs = plt.subplots(1, 2)
            zeros_image = np.zeros_like(sample)
            # replace with nan to have blank background
            zeros_image[:] = np.nan
            for step in chunks:
                # total_structure_mask = np.zeros_like(sample)
                # fig, axs = plt.subplots(1, 2)
                # plt.title(f'step: {step}')

                step_start = step[0]
                step_end = step[1]

                new_img_mask = np.zeros_like(sample)
                # create the mask for finding structures
                # for first version use absolute values of the inputs
                # ranked indices starts from min value
                indices_len = len(ranked_indices)
                for (x, y) in ranked_indices[int(step_start*indices_len):int(step_end*indices_len)]:
                    new_img_mask[x, y] = 1

                total_structure_mask += new_img_mask*step_end

                original_selected_structures = new_img_mask * np.array(sample)
                struct_labels, num_features = label(np.array(new_img_mask))

                # result_dict = {'area': [], 'length': [], 'height': [], 'original_abs_sum': [], 'original_sum': [], 'shap_abs_sum': [], 'shap_sum': [], 'sample_idx': [], 'channel': [], 'uvw': []}
                # calculate all the variables now, later can filter by minimum pixel count - better to collect all data and remove some after than not have at all
                # zeros_image = np.zeros((208, 416))
                # zeros_image = np.zeros_like(struct_labels)
                for struct_idx in range(1, num_features+1):
                    area = np.sum(struct_labels == struct_idx)
                    if area > 13:
                        zeros_image[struct_labels == struct_idx] = step_end

        cmap = plt.cm.RdBu_r  # Use any desired colormap
        cmap.set_bad(color='white')  # Set NaN values to appear as white
        plt.close()
        fig = plt.figure()
        axs = ImageGrid(fig, 111,
                        nrows_ncols=(1, 2),
                        cbar_location="right",
                        direction='row',
                        cbar_mode='single',
                        cbar_pad=0.2,
                        axes_pad=(0.1, 0.05)
                        )
        im1 = axs[0].imshow(total_structure_mask, cmap=cmap)
        axs[0].set_title('Segmentation into deciles')
        cbar = plt.colorbar(im1, cax=axs.cbar_axes[0])
        # increase sbar font
        cbar.ax.tick_params(labelsize=12)
        custom_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        cbar.set_ticks(custom_ticks)
        im2 = axs[1].imshow(zeros_image, cmap=cmap)
        axs[1].set_title('Structures selected by area, $S^+ > 30^2 $')

        # ad xlabel
        axs[0].set_xlabel(r'$x$', fontsize=20)
        axs[1].set_xlabel(r'$x$', fontsize=20)
        axs[0].set_ylabel(r'$z$', fontsize=20)
        # set cbar title on the bottom


        plt.savefig(f'./decile_plot_{u_v_w}_{channel}_sample_0.png', bbox_inches='tight', dpi=300)
        k = 0

k = 0