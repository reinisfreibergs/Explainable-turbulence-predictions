import pandas
from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import platform
import chart_studio.plotly as py
import plotly.figure_factory as ff
from scipy.stats import gaussian_kde
if platform.system() == 'Windows':
    import matplotlib
    matplotlib.use("qt5agg")
import scipy
from tqdm import tqdm
from matplotlib.colors import LogNorm
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
plt.rcParams['figure.figsize'] = (3.5, 2.5)
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
plt.rc('text', usetex=True)

plt.rcParams['font.size'] = 10 + 2
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# Axes and labels
plt.rcParams['axes.labelsize'] = 10 + 2
plt.rcParams['xtick.labelsize'] = 8 + 2
plt.rcParams['ytick.labelsize'] = 8 + 2
plt.rcParams['lines.linewidth'] = 1.0 + 0.2
# Legends and titles
plt.rcParams['legend.fontsize'] = 8 + 2
plt.rcParams['axes.titlesize'] = 12 + 2
def create_contour_plot(X, Y, direction='z'):



    X = (X - np.mean(X)) / np.std(X)
    input_names = [r'\tau_{wx}', r'\tau_{wz}', r'p_{w}']
    direction_names = [r'\tau_{wx}', r'\tau_{wz}', r'p_{w}']
    # Create grid points for KDE
    xmin, xmax = 0.99*X.min(), 1.01*X.max()
    ymin, ymax = 0.99*Y.min(), 1.01*Y.max()
    bins = (1000, 1000)  # Number of bins for the histogram
    H_init, xedges, yedges = np.histogram2d(X, Y, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
    H = scipy.ndimage.gaussian_filter(H_init, sigma=2)
    H = H * (H > 10)

    # recalculate x_min and max from H
    xmin1, xmax1 = xedges[np.where(H)[0].min()], xedges[np.where(H)[0].max()]
    ymin1, ymax1 = yedges[np.where(H)[1].min()], yedges[np.where(H)[1].max()]


    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    X_grid, Y_grid = np.meshgrid(xcenters, ycenters)

    plt.figure()
    im = plt.imshow(np.log10(H.T), origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='viridis', aspect='auto', vmax=4)
    plt.xlim(xmin1, xmax1)
    plt.ylim(ymin1, ymax1)

    cbar = plt.colorbar(im)
    cbar.set_ticks([1, 2, 3, 4])
    def log_format(x, pos):
        return r'$10^{{{:.0f}}}$'.format(x)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(log_format))
    cbar.ax.tick_params(labelsize=10)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
    cbar.ax.set_title('point count', fontsize=10)

    if direction == 'z':
        plt.xlabel(r'$\frac{\tau_{wz} - \overline{\tau_{wz}}}{\sigma_{\tau_{wz}}}$', fontsize=14)
    else:
        plt.xlabel(r'$\frac{\tau_{wx} - \overline{\tau_{wx}}}{\sigma_{\tau_{wx}}}$', fontsize=14)

    plt.ylabel(r'$\frac{p_{w} - \overline{p_{w}}}{\sigma_{p_{w}}}$', fontsize=14)
    # plt.tick_params(labelsize=18)


    # ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune=None))
    # plt.locator_params(axis='y', nbins=5)
    plt.tight_layout()
    if direction == 'z':
        plt.savefig(f'./distribution_p_tau_z.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'./distribution_p_tau_x.png', dpi=300, bbox_inches='tight')
    # plt.show()

    k = 0
    plt.close()


yp = 15
data_shap = np.load(rf'final_shap_{yp}.npy')
data_x = np.load(rf'./x_test_{yp}.npy')
data_y = np.load(rf'./y_test_{yp}.npy')


fig, axs = plt.subplots(figsize=(12, 12))


# create a simple collection of just input samples to create p-tau_x and p-tau_z plots
reshaped_uvw = []
for uvw in range(3):
    reshaped_uvw.append(data_x[:, uvw, :, :].reshape(-1))

# input X, Y
create_contour_plot(reshaped_uvw[1], reshaped_uvw[2], direction='z')

create_contour_plot(reshaped_uvw[0], reshaped_uvw[2], direction='x')


k = 0
