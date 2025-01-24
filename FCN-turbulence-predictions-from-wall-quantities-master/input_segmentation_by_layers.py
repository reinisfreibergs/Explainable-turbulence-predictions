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

yp = 15
data_shap = np.load(rf'final_shap_{yp}.npy')
data_x = np.load(rf'./x_test_{yp}.npy')
data_y = np.load(rf'./y_test_{yp}.npy')

# mean over the 100 samples - (3, 3, 208, 208)
mean_shap = np.mean(np.abs(data_shap), axis=1)
mean_x = np.mean(data_x, axis=0)

result_dict = {'area': [], 'length': [], 'height': [], 'original_abs_sum': [], 'original_sum': [],
                    'shap_abs_sum': [], 'shap_sum': [], 'sample_idx': [], 'channel': [], 'uvw': [],  'output_abs_sum': [], 'output_sum': [],
               'mean_minimum_dist_to_wall': [], 'step_start': []}


k = 0
for u_v_w in range(3):
    for channel in range(3):
        # for sample_idx in tqdm(range(len(data_x))):
        for sample_idx in tqdm(range(100)):


            sample = data_x[sample_idx][channel][8:-8, 8:-8]
            sample_output = data_y[sample_idx][u_v_w]

            sample_shap = data_shap[u_v_w][sample_idx][channel][8:-8, 8:-8]
            # ranked_indices = np.array(np.unravel_index(np.argsort(np.abs(sample).flatten()), sample.shape)).T

            # ranked_indices = np.array(np.unravel_index(np.argsort(np.abs(sample).flatten()), sample.shape)).T

            ranked_indices = np.array(np.unravel_index(np.argsort(np.abs(sample_shap).flatten()), sample_shap.shape)).T

            # ranked_indices_shap = np.array(np.unravel_index(np.argsort(np.abs(data_shap)[u_v_w][sample_idx][channel].flatten()),
            #                                            data_shap[u_v_w][sample_idx][channel].shape)).T

            new_img = np.zeros_like(sample)
            new_img_shap = np.zeros_like(sample)

            total_structure_mask = np.zeros_like(sample)

            n = 10
            points = np.linspace(0, 1, n + 1)
            # Create the chunks by pairing consecutive points
            chunks = [[points[i], points[i + 1]] for i in range(n)]

            for step in chunks:
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
                # zeros_image_2 = np.zeros_like(zeros_image)
                for struct_idx in range(1, num_features+1):
                    area = np.sum(struct_labels == struct_idx)
                    if area > 13:
                        # find the length as the maximum positive value along x axis
                        x, y = np.where(struct_labels == struct_idx)
                        max_x, min_x, max_y, min_y = np.max(x), np.min(x), np.max(y), np.min(y)
                        # add one since have to include the last - if there's one pixel max=min but the length is 1
                        length = max_x - min_x + 1
                        height = max_y - min_y + 1
                        # calculate the original sum of variables over the found structure
                        individual_orig_struct = (struct_labels == struct_idx) * np.array(sample)
                        individual_orig_struct_shap = (struct_labels == struct_idx) * np.array(sample_shap)
                        individual_orig_struct_output = (struct_labels == struct_idx) * np.array(sample_output)
                        original_sum = np.sum(individual_orig_struct)
                        # for now take abs values - otherwise if symmetric around zero will cancel out
                        original_abs_sum = np.sum(np.abs(individual_orig_struct))

                        shap_abs_sum = np.sum(np.abs(individual_orig_struct_shap))
                        shap_sum = np.sum(individual_orig_struct_shap)

                        output_abs_sum = np.sum(np.abs(individual_orig_struct_output))
                        output_sum = np.sum(individual_orig_struct_output)

                        mean_minimum_dist_to_wall = np.mean(np.min(np.array([x, y]), axis=0))
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
                        result_dict['output_abs_sum'].append(output_abs_sum)
                        result_dict['output_sum'].append(output_sum)
                        result_dict['sample_idx'].append(sample_idx)
                        result_dict['channel'].append(channel)
                        result_dict['uvw'].append(u_v_w)
                        result_dict['mean_minimum_dist_to_wall'].append(mean_minimum_dist_to_wall)
                        result_dict['step_start'].append(step_start)
            k = 0


# np.save(f'structure_stats_abs_inputs_yp{yp}.npy', result_dict_full)
# save the full result dict as pd
df = pd.DataFrame(result_dict)
# df.to_csv(f'structure_stats_raw_inputs_yp{yp}_top10_min10.csv')
# df.to_csv(f'structure_stats_abs_inputs_yp{yp}_top20_with_outputs_ranked_by_shap.csv')

k = 0