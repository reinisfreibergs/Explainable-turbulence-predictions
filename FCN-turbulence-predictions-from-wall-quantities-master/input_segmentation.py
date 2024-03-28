import numpy as np
import os
import matplotlib.pyplot as plt
import platform
if platform.system() == 'Windows':
    import matplotlib
    matplotlib.use("qt5agg")
from matplotlib.ticker import ScalarFormatter
from skimage.transform import resize


data_shap = np.load(r'final_shap_15_older.npy')
data_x = np.load(r'./x_test_15.npy')
data_y = np.load(r'./y_test_15.npy')

# mean over the 100 samples - (3, 3, 208, 208)
mean_shap = np.mean(np.abs(data_shap), axis=1)
mean_x = np.mean(data_x, axis=0)

sample_idx = 0
channel = 0

sample = data_x[sample_idx][channel]
ranked_indices = np.array(np.unravel_index(np.argsort(sample.flatten()), sample.shape)).T
ranked_indices_shap = np.array(np.unravel_index(np.argsort(np.abs(data_shap)[0][sample_idx][channel].flatten()),
                                           data_shap[0][sample_idx][channel].shape)).T

new_img = np.zeros_like(sample)
new_img_shap = np.zeros_like(sample)

# top10
for (x, y) in ranked_indices[int(0.9*len(ranked_indices)):]:
    new_img[x, y] = 1

for (x, y) in ranked_indices[:int(0.1*len(ranked_indices))]:
    new_img[x, y] = 1

for (x_s, y_s) in ranked_indices_shap[int(0.9*len(ranked_indices_shap)):]:
    new_img_shap[x_s, y_s] = 2


plt.imshow(new_img)
plt.imshow(new_img_shap, alpha=0.6)
# plt.show()

k = 0