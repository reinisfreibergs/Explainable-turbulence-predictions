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

'''
..
'''


yp = 15
data_shap = np.load(rf'final_shap_{yp}.npy')
data_x = np.load(rf'./x_test_{yp}.npy')
data_y = np.load(rf'./y_test_{yp}.npy')


fig = plt.figure()  # Adjust the figure size as needed
grid = ImageGrid(fig, 111,          # similar to subplot(111)
                 nrows_ncols=(3, 1), # creates a 3x1 grid
                    axes_pad=0.1,
                 cbar_mode='each',   # one colorbar for each image
                 cbar_location='right',
                 cbar_pad=0.1        # pad between image and colorbar
                 )

# Display each image in the grid
image1 = data_x[0][0]
image2 = data_x[0][1]
image3 = data_x[0][2]
for i, (ax, img) in enumerate(zip(grid, [image1, image2, image3])):
    # im = ax.imshow(resize(img, (208, 416)), cmap='RdBu_r')\
    data = resize(img, (208, 416))
    im = ax.imshow(data, cmap='RdBu', origin='lower', extent=[-2 * np.pi, 2 * np.pi, -np.pi, np.pi])
    # cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    # cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
    # cbar.set_ticks([np.min(data), np.max(data)])  # Set ticks for the colorbar

    if i < 1:
        ax.set_xticks([])
        ax.set_xticklabels([])
    ax.cax.colorbar(im)

k = 0