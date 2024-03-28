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


input_names = [r'$\tau_{wx}$', r'$\tau_{wz}$', r'$p_{w}$']
direction_names = ['u', 'v', 'w']

# Create a figure and axes
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(24, 16))  # Adjust figsize as needed
for i in range(3):
    for j in range(4):
        ax = axes[i, j]

        # domain size is (4pi*h, h, 2pi*h)
        if j == 0:
            data = resize(mean_x[i], (208, 416))
            ax.set_title(fr'Avg input {input_names[i]}', fontsize=24)
            ax.set_ylabel(r'$\frac{z}{h}$', fontsize=24)
        else:
            data = resize(mean_shap[j-1][i], (208, 416))
            ax.set_title(fr'Avg |SHAP({direction_names[j-1]},{input_names[i]})|', fontsize=24)

        if i == 2:
            ax.set_xlabel(r'$\frac{x}{h}$', fontsize=24)

        im = ax.imshow(data, cmap='RdBu', origin='lower', extent=[-2*np.pi, 2*np.pi, -np.pi, np.pi])
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        cbar.set_ticks([np.min(data), np.max(data)])  # Set ticks for the colorbar
        cbar.ax.set_yticklabels(['{:.2e}'.format(np.min(data)), '{:.2e}'.format(np.max(data))])  # Set tick labels

plt.tight_layout()
plt.show()

# cmap='coolwarm'
k = 0