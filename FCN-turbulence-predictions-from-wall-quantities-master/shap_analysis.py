import numpy as np
import os
import matplotlib.pyplot as plt
import platform
if platform.system() == 'Windows':
    import matplotlib
    matplotlib.use("qt5agg")
from matplotlib.ticker import ScalarFormatter


data_shap = np.load(r'final_shap_100_v3.npy')
data_x = np.load(r'./x_test_100.npy')

# mean over the 100 samples - (3, 3, 208, 208)
mean_shap = np.mean(np.abs(data_shap), axis=1)
mean_x = np.mean(data_x, axis=0)


input_names = [r'$\tau_{wx}$', r'$\tau_{wz}$', r'$p_{w}$']
direction_names = ['u', 'v', 'w']

# Create a figure and axes
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 9))  # Adjust figsize as needed
for i in range(3):
    for j in range(4):
        ax = axes[i, j]
        if j == 0:
            data = mean_x[i]
            ax.set_title(fr'Avg input {input_names[i]}')
        else:
            data = mean_shap[i][j-1]
            ax.set_title(fr'Avg |SHAP({direction_names[j-1]},{input_names[i]})|')

        im = ax.imshow(data, cmap='coolwarm')
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
        cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        cbar.set_ticks([np.min(data), np.max(data)])  # Set ticks for the colorbar
        cbar.ax.set_yticklabels(['{:.2e}'.format(np.min(data)), '{:.2e}'.format(np.max(data))])  # Set tick labels

plt.tight_layout()

# cmap='coolwarm'
k = 0