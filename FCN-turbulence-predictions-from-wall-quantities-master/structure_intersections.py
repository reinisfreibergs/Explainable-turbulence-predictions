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

'''
Calculate how many of the top 10% of the most important pixels in the input image are also in the top 10% of the most important pixels in the shap image
'''

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

yp = 15
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
        # for sample_idx in tqdm(range(len(data_x))):
        score = []
        for sample_idx in range(len(data_x)):

            sample = data_x[sample_idx][channel][8:-8, 8:-8]
            sample_shap = data_shap[u_v_w][sample_idx][channel][8:-8, 8:-8]
            sample_output = data_y[sample_idx][u_v_w]
            ranked_indices_output = np.array(np.unravel_index(np.argsort(np.abs(sample_output).flatten()), sample.shape)).T
            ranked_indices = np.array(np.unravel_index(np.argsort(np.abs(sample).flatten()), sample.shape)).T
            # ranked_indices = np.array(np.unravel_index(np.argsort(np.abs(sample).flatten()), sample.shape)).T
            ranked_indices_shap = np.array(np.unravel_index(np.argsort(np.abs(sample_shap).flatten()),
                                                       sample_shap.shape)).T

            new_img = np.zeros_like(sample)
            new_img_shap = np.zeros_like(sample)

            PLOT_SAMPLE = False
            if PLOT_SAMPLE:
                # top10
                for (x, y) in ranked_indices[int(0.9*len(ranked_indices)):]:
                    new_img[x, y] += 1

                # for (x, y) in ranked_indices[:int(0.1*len(ranked_indices))]:
                #     new_img[x, y] += 1

                for (x_s, y_s) in ranked_indices_shap[int(0.8*len(ranked_indices_shap)):]:
                    new_img[x_s, y_s] += 2

                plt.imshow(new_img)
                # plt.imshow(new_img_shap, alpha=0.6)


            shap_list = ranked_indices_shap[int(0.8*len(ranked_indices_shap)):]
            input_list = ranked_indices[int(0.8*len(ranked_indices)):]
            output_list = ranked_indices_output[int(0.8*len(ranked_indices_output)):]

            # Convert the numpy arrays to sets of tuples
            set1 = set(map(tuple, input_list))
            # set1 = set(map(tuple, output_list))
            set2 = set(map(tuple, shap_list))

            # Find the intersection of the two sets
            intersection = set1.intersection(set2)
            intersection_len = len(intersection)
            score.append(intersection_len)
            k = 0

        print(f'channel: {channel} uvw: {u_v_w} score: {np.mean(score)}, score_std: {np.std(score)}, relative_score: {np.mean(score)/len(input_list)}, relative_score_std: {np.std(score)/len(input_list)}')