import numpy as np
import os
import matplotlib.pyplot as plt
import platform
if platform.system() == 'Windows':
    import matplotlib
    matplotlib.use("qt5agg")
from matplotlib.ticker import ScalarFormatter
from skimage.transform import resize
from scipy.ndimage import label
from tqdm import tqdm
import pandas as pd
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.ticker import MaxNLocator, ScalarFormatter, FormatStrFormatter, FuncFormatter
matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)

yp = 100
data_shap = np.load(rf'final_shap_{yp}.npy')
data_x = np.load(rf'./x_test_{yp}.npy')
data_y = np.load(rf'./y_test_{yp}.npy')

# mean over the 100 samples - (3, 3, 208, 208)
mean_shap = np.mean(np.abs(data_shap), axis=1)
mean_x = np.mean(data_x, axis=0)

result_dict = {'area': [], 'length': [], 'height': [], 'original_abs_sum': [], 'original_sum': [],
                    'shap_abs_sum': [], 'shap_sum': [], 'sample_idx': [], 'channel': [], 'uvw': []}


for u_v_w in range(3):
    for channel in range(3):
        for sample_idx in tqdm(range(len(data_x))):
        # for sample_idx in tqdm(range(100)):

            sample = data_x[sample_idx][channel]

            sample_shap = data_shap[u_v_w][sample_idx][channel]
            # ranked_indices = np.array(np.unravel_index(np.argsort(np.abs(sample).flatten()), sample.shape)).T
            ranked_indices = np.array(np.unravel_index(np.argsort((sample).flatten()), sample.shape)).T
            # ranked_indices_shap = np.array(np.unravel_index(np.argsort(np.abs(data_shap)[u_v_w][sample_idx][channel].flatten()),
            #                                            data_shap[u_v_w][sample_idx][channel].shape)).T

            new_img = np.zeros_like(sample)
            new_img_shap = np.zeros_like(sample)

            PLOT_SAMPLE = False
            if PLOT_SAMPLE:
                sample_output = data_y[sample_idx][channel]
                ranked_indices_output = np.array(np.unravel_index(np.argsort((sample_output).flatten()), sample.shape)).T
                # top10
                for (x, y) in ranked_indices[int(0.9*len(ranked_indices_output)):]:
                    new_img[x, y] = 1

                for (x, y) in ranked_indices[:int(0.1*len(ranked_indices_output))]:
                    new_img[x, y] = 1

                for (x_s, y_s) in ranked_indices_shap[int(0.9*len(ranked_indices_shap)):]:
                    new_img_shap[x_s, y_s] = 2

                plt.imshow(new_img)
                plt.imshow(new_img_shap, alpha=0.6)
                k = 0
                # plt.show()


            new_img_mask = np.zeros_like(sample)
            # create the mask for finding structures
            # for first version use absolute values of the inputs
            # ranked indices starts from min value
            for (x, y) in ranked_indices[int(0.8*len(ranked_indices)):]:
                new_img_mask[x, y] = 1

            for (x, y) in ranked_indices[:int(0.2*len(ranked_indices))]:
                new_img_mask[x, y] = 1


            original_selected_structures = new_img_mask * np.array(sample)
            struct_labels, num_features = label(np.array(new_img_mask))

            # result_dict = {'area': [], 'length': [], 'height': [], 'original_abs_sum': [], 'original_sum': [], 'shap_abs_sum': [], 'shap_sum': [], 'sample_idx': [], 'channel': [], 'uvw': []}
            # calculate all the variables now, later can filter by minimum pixel count - better to collect all data and remove some after than not have at all
            zeros_image = np.zeros((208, 416))
            zeros_image_2 = np.zeros_like(zeros_image)
            for struct_idx in range(1, num_features+1):
                area = np.sum(struct_labels == struct_idx)
                # find the length as the maximum positive value along x axis
                x, y = np.where(struct_labels == struct_idx)
                max_x, min_x, max_y, min_y = np.max(x), np.min(x), np.max(y), np.min(y)
                # add one since have to include the last - if there's one pixel max=min but the length is 1
                length = max_x - min_x + 1
                height = max_y - min_y + 1
                # calculate the original sum of variables over the found structure
                individual_orig_struct = (struct_labels == struct_idx) * np.array(sample)
                individual_orig_struct_shap = (struct_labels == struct_idx) * np.array(sample_shap)
                original_abs_sum = np.sum(individual_orig_struct)
                # for now take abs values - otherwise if symmetric around zero will cancel out
                original_sum = np.sum(np.abs(individual_orig_struct))

                shap_abs_sum = np.sum(np.abs(individual_orig_struct_shap))
                shap_sum = np.sum(individual_orig_struct_shap)

                # if area > 20:
                #     zeros_image = zeros_image + resize(individual_orig_struct, (208, 416))
                #     zeros_image_2 = zeros_image_2 + resize((1 * (individual_orig_struct != 0)) * shap_abs_sum, (208, 416))

                result_dict['area'].append(area)
                result_dict['length'].append(length)
                result_dict['height'].append(height)
                result_dict['original_abs_sum'].append(original_abs_sum)
                result_dict['original_sum'].append(original_sum)
                result_dict['shap_abs_sum'].append(shap_abs_sum)
                result_dict['shap_sum'].append(shap_sum)
                result_dict['sample_idx'].append(sample_idx)
                result_dict['channel'].append(channel)
                result_dict['uvw'].append(u_v_w)

            CREATE_STRUCTURE_PLOTS = False
            if CREATE_STRUCTURE_PLOTS:
                original_input = resize(np.array(sample), (208, 416))
                original_shap_input = resize(np.array(np.abs(sample_shap)), (208, 416))
                selected_input_structures = resize(np.array(zeros_image), (208, 416))
                zeros_image_2[np.where(zeros_image_2 == 0)] = -0.002
                selected_shap_structures = resize(np.array(zeros_image_2), (208, 416))


                fig = plt.figure(figsize=(24, 16))
                axs = ImageGrid(fig, 111,
                                nrows_ncols=(2, 2),
                                cbar_location="right",
                                cbar_mode='each',
                                direction='row',
                                cbar_size="5%",
                                cbar_pad=0.2,
                                share_all=True,
                                axes_pad=(0.8, 0.5)
                                )

                vmin, vmax = np.min(original_input), np.max(original_input)

                vmin3, vmax3 = np.min(selected_shap_structures), np.max(selected_shap_structures)


                im1 = axs[0].imshow(original_input, cmap='RdBu_r', vmin=vmin,
                                    vmax=vmax, extent=[-2 * np.pi, 2 * np.pi, -np.pi, np.pi])

                im2 = axs[1].imshow(selected_input_structures, cmap='RdBu_r',
                                    vmin=vmin, vmax=vmax,
                                    extent=[-2 * np.pi, 2 * np.pi, -np.pi, np.pi])  # modified input

                im3 = axs[2].imshow(original_shap_input, cmap='RdBu_r',
                                    extent=[-2 * np.pi, 2 * np.pi, -np.pi, np.pi])  # original prediction

                im4 = axs[3].imshow(selected_shap_structures, cmap='RdBu_r', vmin=vmin3, vmax=vmax3,
                                    extent=[-2 * np.pi, 2 * np.pi, -np.pi, np.pi])

                axs[0].set_title(r'$\tau_{wx}$ input', fontsize=24)
                axs[1].set_title(r'Segmented $\tau_{wx}$ input', fontsize=24)
                axs[2].set_title(r'|SHAP($u$, $\tau_{wx}$)|', fontsize=24)
                axs[3].set_title(r'|SHAP($u$, $\tau_{wx}$)| sum per $\tau_{wx}$ segment', fontsize=20)

                axs[0].set_ylabel(r'$z/h$', fontsize=24)
                axs[2].set_ylabel(r'$z/h$', fontsize=24)
                axs[2].set_xlabel(r'$x/h$', fontsize=24)
                axs[3].set_xlabel(r'$x/h$', fontsize=24)

                cbar1 = plt.colorbar(im1, cax=axs.cbar_axes[0], format=FormatStrFormatter('%.3f'))
                cbar2 = plt.colorbar(im2, cax=axs.cbar_axes[1], format=FormatStrFormatter('%.3f'))
                cbar3 = plt.colorbar(im3, cax=axs.cbar_axes[2])
                cbar4 = plt.colorbar(im4, cax=axs.cbar_axes[3])
                # cbar1.ax.tick_params(labelsize=16)
                # cbar2.ax.tick_params(labelsize=16)
                # cbar3.ax.tick_params(labelsize=16)


                images = [original_input, selected_input_structures, original_shap_input, selected_shap_structures]
                for idx, cbar in enumerate([cbar1, cbar2, cbar3, cbar4]):
                    min_val = np.min(images[idx])
                    max_val = np.max(images[idx])
                    ticks = np.linspace(min_val, max_val, 5)

                    cbar.set_ticks(ticks)

                    if idx < 2:  # Upper plots
                        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.3f}'))
                    else:  # Lower plots


                        # cbar.update_ticks()
                        # cbar.ax.yaxis.set_major_formatter(cbar.formatter)
                        # cbar.formatter = FuncFormatter(lambda x, pos: f'{x:1.1e}')
                        # cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2e}'))
                        # cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
                        cbar.formatter = ScalarFormatter(useMathText=True)

                        cbar.formatter.set_scientific(True)
                        cbar.formatter.set_powerlimits((-2, 2))

                        for t in cbar.ax.get_yticklabels():
                            t.set_text(cbar.ax.get_yticklabels()[0].get_text()[:-4] + cbar.ax.get_yticklabels()[0].get_text()[-2:])  # Removes extra zeros
                        cbar.ax.set_yticklabels(cbar.ax.get_yticklabels())
                        cbar.update_ticks()

                    cbar.ax.tick_params(labelsize=14)
                    cbar.update_ticks()

                plt.savefig(
                    f'./images/input_segmentation_side_by_side_1.png', dpi=1300, bbox_inches='tight')

                k = 0


# np.save(f'structure_stats_abs_inputs_yp{yp}.npy', result_dict_full)
# save the full result dict as pd
df = pd.DataFrame(result_dict)
df.to_csv(f'structure_stats_raw_inputs_yp{yp}_top10_min10.csv')

k = 0