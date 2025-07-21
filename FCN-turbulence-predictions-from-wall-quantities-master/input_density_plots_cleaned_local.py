import pandas
import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
import platform
if platform.system() == 'Windows':
    import matplotlib
    matplotlib.use("qt5agg")
import scipy
from tqdm import tqdm
from matplotlib.colors import LogNorm
from matplotlib import ticker
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
def create_contour_plot(X, Y, show_mean=False, save=False, uvw=0, channel=0, name='output', return_x_y = False):

    if not return_x_y:

        X = (X - np.mean(X)) / np.std(X)
        input_names = [r'\tau_{wx}', r'\tau_{wz}', r'p_{w}']
        direction_names = ['u', 'v', 'w']
        # Create grid points for KDE
        xmin, xmax = 0.99*X.min(), 1.01*X.max()
        ymin, ymax = 0.99*Y.min(), 1.01*Y.max()

        if yp==50:
            xmin = np.percentile(X, 0.001)
            xmax = np.percentile(X, 99.999)
            ymax = np.percentile(Y, 99.999)

        bins = (1000, 1000)  # Number of bins for the histogram
        H_init, xedges, yedges = np.histogram2d(X, Y, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
        H_init_filtered = scipy.ndimage.gaussian_filter(H_init, sigma=2)
        H = H_init_filtered * (H_init_filtered > 10)

        # recalculate x_min and max from H
        xmin1, xmax1 = xedges[np.where(H)[0].min()], xedges[np.where(H)[0].max()]
        ymin1, ymax1 = yedges[np.where(H)[1].min()], yedges[np.where(H)[1].max()]

        # xedges = xedges[np.where(H)[0].min() : np.where(H)[0].max()]
        # yedges = yedges[np.where(H)[1].min() : np.where(H)[1].max()]

        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        X_grid, Y_grid = np.meshgrid(xcenters, ycenters)

        plt.figure()
        im = plt.imshow(np.log10(H.T), origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='viridis', aspect='auto')

        # contour = plt.contour(X_grid, Y_grid, H.T, levels=levels, colors='red', linewidths=0.2, extent=[xmin, xmax, ymin, ymax], norm=LogNorm())
        # contourf = plt.contourf(X_grid, Y_grid, H.T, levels=levels, extent=[xmin, xmax, ymin, ymax], norm=LogNorm())
        # contour = plt.contour(X_grid, Y_grid, np.log(H.T), levels=20, colors='red', linewidths=0.2, extent=[xmin, xmax, ymin, ymax])
        # plt.figure()
        # contourf = plt.contourf(X_grid, Y_grid, np.log10(H.T), levels=25, extent=[xmin, xmax, ymin, ymax])
        # plt.contourf(X_grid, Y_grid, make_zeros_inf(np.log10(np.array(H.T)) * (np.log10(np.array(H.T)) > -1)), levels=25,
        #              extent=[xmin, xmax, ymin, ymax])
        cbar = plt.colorbar(im)
        cbar.set_ticks([1, 2, 3, 4, 5])
        def log_format(x, pos):
            return r'$10^{{{:.0f}}}$'.format(x)
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(log_format))
        cbar.ax.tick_params(labelsize=10)

        ax = plt.gca()
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(True)
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
        cbar.ax.set_title('point count', fontsize=10)

        if name == 'input':
            k = 0
            # plt.xlabel(rf'${input_names[channel]}$')
            plt.xlabel(r'$\frac{\tau_{wx} - \overline{\tau_{wx}}}{\sigma_{\tau_{wx}}}$', fontsize=14)
            # plt.xlabel(
            #     rf'$\frac{{{input_names[channel]} - \overbar{input_names[channel]}}}{{\overbar{{input_names[channel]}}}}}$')
            # plt.xlabel(
            #     rf'$\frac{{{input_names[channel]} - \text{{mean(}} {input_names[channel]})}}{{\text{{std(}} {input_names[channel]})}}$')
            # plt.xlabel(rf'${input_names[channel]}$', fontsize=24)
            # plt.title(
            #     f'${input_names[channel]}$ input SHAP value distribution for the ${direction_names[uvw]}$ fluctuation \n',
            #     fontsize=20)
        else:
            plt.xlabel(rf'${direction_names[uvw]}$')
            # plt.xlabel(rf'${direction_names[uvw]}$', fontsize=24)
            # plt.title(
            #     f'${input_names[channel]}$ input SHAP value distribution for the ${direction_names[uvw]}$ fluctuation \n',
            #     fontsize=20)

        plt.ylabel(rf"$\phi_{{{direction_names[uvw]}, {input_names[channel]}}}$", fontsize=14)
        if NORMALIZE_SHAP:
            plt.ylabel(rf"$\widehat{{\phi}}_{{{direction_names[uvw]}, {input_names[channel]}}}$", fontsize=14)
        # plt.ylabel(rf'$\phi({direction_names[uvw]}, {input_names[channel]})$', fontsize=24)
        # plt.tick_params(labelsize=18)


        # ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune=None))
        # plt.locator_params(axis='y', nbins=5)
        plt.tight_layout()
        # plt.savefig(f'./images/{name}_abs_shap_scaled_log_density_{yp}_channel_{channel}_uvw_{uvw}.png',
        #             dpi=200, bbox_inches='tight')
        # plt.show()

        k = 0
        # plt.close()

    if return_x_y:
        xmin = np.percentile(X, 0.001)
        xmax = np.percentile(X, 99.999)
        ymax = np.percentile(Y, 99.99)
        ymin = np.percentile(Y, 0.01)
        hist_rr, xedges, yedges = np.histogram2d(X, Y, bins=1000, range=[[xmin, xmax], [ymin, ymax]])
        hist_r = scipy.ndimage.gaussian_filter(hist_rr, sigma=2)
        # hist = make_zeros_inf(np.log10(hist) * (np.log10(hist) > 1))
        hist = hist_r * (hist_r > 10.5)
        # xedges[np.where(H)[0].min() : np.where(H)[0].max()])
        # yedges[np.where(H)[1].min()], yedges[np.where(H)[1].max()]

        # Calculate the bin centers
        x_bin_centers = (xedges[:-1] + xedges[1:]) / 2
        y_bin_centers = (yedges[:-1] + yedges[1:]) / 2
        # Calculate the top 1% threshold for each bin along the x-axis
        top_1_percent_line = []
        for i in range(len(x_bin_centers)):
            y_counts = hist[i, :]
            cumulative_counts = np.cumsum(y_counts)
            total_counts = cumulative_counts[-1]
            # change the following to switch to other values
            THRESHOLD_VALUE = 0.99
            threshold_value = THRESHOLD_VALUE * total_counts
            top_bin_idx = np.searchsorted(cumulative_counts, threshold_value)
            top_1_percent_y_value = y_bin_centers[top_bin_idx]
            top_1_percent_line.append((x_bin_centers[i], top_1_percent_y_value))

        # Separate x and y values for the top 1% line
        top_1_percent_x, top_1_percent_y = zip(*top_1_percent_line)
        window_length = 21  # Window length must be odd
        polyorder = 2  # Polynomial order
        top_1_percent_y_smooth = savgol_filter(top_1_percent_y, window_length, polyorder)

        minx, maxx = np.where(hist)[0].min(), np.where(hist)[0].max()
        # plt.plot(top_1_percent_x, savgol_filter(top_1_percent_y, window_length, polyorder), color='red', label='Top 1% Threshold')
        # plt.xlim(xmin1, xmax1)
        # plt.ylim(ymin1, ymax1)
        # ax.set_yticks(np.arange(0.2e-4, 1.1e-4, 0.2e-4))
        # plt.savefig(f'./distribution_fixed_{yp}.png', dpi=300, bbox_inches='tight')

        return top_1_percent_x[minx:maxx], top_1_percent_y_smooth[minx:maxx]

    CREATE_COUNT_COMPARISON = False
    if CREATE_COUNT_COMPARISON:
        # create 2 subplots below each other
        fig, axs = plt.subplots(2, figsize=(12, 12))
        axs[0].imshow(H.T, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='viridis', aspect='auto')
        contourf0 = axs[0].contourf(X_grid, Y_grid, np.log(H.T), levels=20, extent=[xmin, xmax, ymin, ymax])
        axs[1].plot(np.linspace(xmin, xmax, num=1000), np.sum(H.T, axis=0))
        # axs[1].plot(np.sum(H.T, axis=1), np.linspace(ymin, ymax, num=1000))
        axs[1].set_xlim([xmin, xmax])

        lowest10 = np.argmax(np.cumsum(np.sum(H.T, axis=0)) / np.sum(H.T) > 0.2)
        highest10 = np.argmax(np.cumsum(np.sum(H.T, axis=0)) / np.sum(H.T) > 0.8)

        highest1_shap = np.argmax(np.cumsum(np.sum(H.T, axis=1)[::-1]) / np.sum(H.T) > 0.01)
        highest10_shap = np.argmax(np.cumsum(np.sum(H.T, axis=1)[::-1]) / np.sum(H.T) > 0.1)
        # axs[0].hlines(y=np.linspace(ymin, ymax, num=1000)[1000-highest1_shap], xmin=xmin, xmax=xmax, color='red')
        # axs[0].hlines(y=np.linspace(ymin, ymax, num=1000)[1000 - highest10_shap], xmin=xmin, xmax=xmax, color='red')
        # axs[1].hlines(y=np.linspace(ymin, ymax, num=1000)[1000 - highest10_shap], xmin=xmin, xmax=xmax, color='red')
        # axs[1].hlines(y=0, xmin=0, xmax=np.sum(H.T), color='red')
        # axs[1].vlines(x=0, ymin=0, ymax=ymax, color='red')



        # axs[0].vlines(x=np.linspace(xmin, xmax, num=1000)[lowest10], ymin=ymin, ymax=ymax, color='red',
        #               linestyles='--', linewidth=1)
        # axs[0].vlines(x=np.linspace(xmin, xmax, num=1000)[highest10], ymin=ymin, ymax=ymax, color='red',
        #               linestyles='--', linewidth=1)
        #
        # axs[1].vlines(x=np.linspace(xmin, xmax, num=1000)[lowest10], ymin=0, ymax=np.max(np.sum(H.T, axis=0)),
        #               color='red',
        #               linestyles='--', linewidth=1)
        # axs[1].vlines(x=np.linspace(xmin, xmax, num=1000)[highest10], ymin=0, ymax=np.max(np.sum(H.T, axis=0)),
        #               color='red',
        #               linestyles='--', linewidth=1)
        axs[0].set_ylabel(rf'$\phi({direction_names[uvw]}, {input_names[channel]})$', fontsize=24)
        axs[0].set_xlabel(rf'${input_names[channel]}$', fontsize=24)
        axs[1].set_ylabel(rf'data point count', fontsize=22)
        axs[1].set_xlabel(rf'${input_names[channel]}$', fontsize=24)
        axs[0].tick_params(labelsize=18)
        axs[1].tick_params(labelsize=18)
        plt.suptitle(rf'Margins by selecting the top 20% highest and lowest values by input ${input_names[channel]}$',
                     fontsize=24)

    k = 0



Ret = 550
yp = 50
# data_shap = np.load(rf'final_shap_{yp}_{Ret}_100.npy')
# data_x = np.load(rf'./data_x_{yp}_{Ret}_100.npy')
# data_y = np.load(rf'./data_y_{yp}_{Ret}_100.npy')
data_shap = np.load(rf'final_shap_{yp}_{Ret}.npy')
data_x = np.load(rf'./data_x_{yp}_{Ret}.npy')
data_y = np.load(rf'./data_y_{yp}_{Ret}.npy')


REGRESSION = False
NORMALIZE_SHAP = True
tops = []
NUM_SAMPLES = data_x.shape[0] # Should be 100 based on data_x shape
CROP = 8
for uvw in range(3):
    for channel in range(3):
        print(f"Processing uvw={uvw}, channel={channel}")
        all_samples_raw_x = data_x[:NUM_SAMPLES, channel, CROP:-CROP, CROP:-CROP]
        means_x = np.mean(all_samples_raw_x, axis=(1, 2), keepdims=True)
        normalized_x = all_samples_raw_x - means_x
        samples_input = normalized_x.reshape(-1)

        all_samples_raw_shap = data_shap[uvw, :NUM_SAMPLES, channel, CROP:-CROP, CROP:-CROP]

        if NORMALIZE_SHAP:
            shap_for_norm = data_shap[uvw, :NUM_SAMPLES, :, CROP:-CROP, CROP:-CROP]

            shap_mean_values_per_sample = np.mean(shap_for_norm, axis=(1, 2, 3))

            normalizers = shap_mean_values_per_sample.reshape(NUM_SAMPLES, 1, 1) + 1e-9

            normalized_shap = all_samples_raw_shap / normalizers
        else:
            # If not normalizing, the result is just the cropped data
            normalized_shap = all_samples_raw_shap

        # Reshape the entire 3D block into a single 1D array
        samples_shap = normalized_shap.reshape(-1)

        colors = ['#440154', '#31688e', '#a0da39']
        line_types = ['-', '--', ':']

        # to create the single plot add return_x_y=False and debug into first part
        top_x, top_y = create_contour_plot(samples_input, np.abs(samples_shap), uvw=uvw, channel=channel, name='input', return_x_y=True)
        tops.append([top_x, top_y])
