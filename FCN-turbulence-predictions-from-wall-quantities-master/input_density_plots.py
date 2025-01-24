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
def create_contour_plot(X, Y, show_mean=False, save=False, uvw=0, channel=0, name='output', return_x_y = False):

    if not return_x_y:

        X = (X - np.mean(X)) / np.std(X)
        input_names = [r'\tau_{wx}', r'\tau_{wz}', r'p_{w}']
        direction_names = ['u', 'v', 'w']
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
        hist_rr, xedges, yedges = np.histogram2d(X, Y, bins=1000)
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
            threshold_value = 0.99 * total_counts
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
        # plt.savefig(f'./distribution_fixed.png', dpi=300, bbox_inches='tight')

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

def linear_fit(X, y, channel, uvw, combined_df):
    coefs_all = None
    # combined_df = pandas.DataFrame()

    # select only those samples where uvw = 0 and channel=0
    # df = df_init[(df_init['uvw'] == uvw) & (df_init['channel'] == channel) & (df_init['area'] > 10)]
    # #
    # # X = df[['area', 'length', 'height', 'original_abs_sum_per_area', 'original_sum_per_area', 'mean_minimum_dist_to_wall', 'output_abs_sum_per_area', 'output_sum_per_area']].to_numpy()
    # X = df[
    #     ['length', 'height', 'original_abs_sum', 'original_sum', 'mean_minimum_dist_to_wall', 'output_abs_sum',
    #      'output_sum']].to_numpy()

    # check if scaling works - it shouldnt matter since its a linear transformation
    # X[:, 0] *= 100

    # X = df[['area']].to_numpy()
    # y = df['shap_abs_sum'].to_numpy()
    # y = df['shap_abs_sum_per_area'].to_numpy()
    # y = (df['shap_abs_sum'] / df['area']).to_numpy()
    # X = np.concatenate()
    # add another column of ones to X

    scale = True
    if scale:
        scaler = preprocessing.StandardScaler()
        # scaler = preprocessing.MinMaxScaler()
        X = scaler.fit_transform(X)

        scaler_y = preprocessing.StandardScaler()
        y = scaler_y.fit_transform(y.reshape(-1, 1))

    # no need to manually add the linear term since it's already included as intercept
    # X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(X, y.squeeze())

    # Predict on the testing set
    y_pred = regr.predict(X)

    # Calculate R-squared score
    r_squared = r2_score(y, y_pred)
    coefs = np.concatenate((regr.coef_, np.array(regr.intercept_)[None]))
    print(f'uvw: {uvw} channel: {channel}')
    print(f'coefs: {coefs}')
    print(f'r^2: {r_squared}')

    if coefs_all is None:
        coefs_all = coefs
        r_squared_all = r_squared
    else:
        coefs_all = np.vstack((coefs_all, coefs))
        r_squared_all = np.vstack((r_squared_all, r_squared))
    k = 0

    df = pandas.DataFrame(np.hstack((r_squared_all, coefs_all)))
    # spacer_df = pandas.DataFrame([f'uvw: {uvw} channel: {channel}'])
    # iteration_df = pandas.concat([df, spacer_df], ignore_index=True)

    combined_df = pandas.concat([combined_df, df], ignore_index=True, axis=1)

    return combined_df

def clustering(features, shap_features):

    combined_features = np.column_stack((features, shap_features))[None, :]

    # Step 6: Apply clustering (K-means example)
    flattened_combined_features = combined_features.reshape(-1, combined_features.shape[2])
    kmeans = KMeans(n_clusters=5, random_state=0).fit(flattened_combined_features)
    labels = kmeans.labels_
    labels_reshaped = labels.reshape(1, 192, 192)
    # labels.reshape(200, 192, 192)[0]
    # for cluster in range(5):
    #     cluster_indices = np.where(labels_reshaped == cluster)
    #     cluster_shap_values = central_shap_values[cluster_indices]
    #
    #     # Visualize mean SHAP values for the cluster
    #     mean_shap = np.mean(cluster_shap_values, axis=0)
    #
    #     plt.figure()
    #     plt.imshow(mean_shap.mean(axis=0), cmap='viridis')  # Taking mean across channels
    #     plt.title(f'Cluster {cluster}')
    #     plt.colorbar()
    #     plt.show()
    k = 0
    # Reshape labels to the original shape for analysis
    # labels_reshaped = labels.reshape(images.shape[0], n_regions_x, n_regions_y)

    # # Step 7: Analyze and visualize clusters
    # for cluster in range(5):
    #     cluster_indices = np.where(labels_reshaped == cluster)
    #     cluster_shap_values = central_shap_values[cluster_indices]
    #
    #     # Visualize mean SHAP values for the cluster
    #     mean_shap = np.mean(cluster_shap_values, axis=0)
    #
    #     plt.figure()
    #     plt.imshow(mean_shap.mean(axis=0), cmap='viridis')  # Taking mean across channels
    #     plt.title(f'Cluster {cluster}')
    #     plt.colorbar()
    #     plt.show()

    # return inputs

yp = 15
data_shap = np.load(rf'final_shap_{yp}.npy')
# data_shap = np.load(rf'final_shap_{yp}_gaussian_blur_7.npy')
# data_shap = np.load(rf'final_shap_{yp}_time_delay_10_steps.npy')
# data_shap = np.load(rf'final_shap_{yp}_time_delay_+-50_10_steps.npy')
# data_shap = np.load(rf'final_shap_{yp}_random_10.npy')
# data_shap = np.load(rf'final_shap_{yp}_random_4_abs.npy')
# data_shap = np.load(rf'final_shap_{yp}_75_percentile.npy')
# data_shap = np.load(rf'final_shap_{yp}_gaussian_15.npy')
# data_shap = np.load(rf'final_shap_{yp}_blur_5_15_21_35_51_.npy')
# data_shap = np.load(rf'final_shap_{yp}_blur_5_15_35_51_2.npy')
# data_shap = np.load(rf'final_shap_{yp}_random_4_correct.npy')
# data_shap = np.load(rf'final_shap_{yp}_blur_5_9_15_35.npy')
data_x = np.load(rf'./x_test_{yp}.npy')
data_y = np.load(rf'./y_test_{yp}.npy')


REGRESSION = False
# input_names = [r'$\tau_{wx}$', r'$\tau_{wz}$', r'$p_{w}$']
# direction_names = ['u', 'v', 'w']
input_names = [r'\tau_{wx}', r'\tau_{wz}', r'p_{w}']
direction_names = ['u', 'v', 'w']
fig, axs = plt.subplots(figsize=(12, 12))


NORMALIZE_SHAP = True
tops = []
for uvw in range(3):
    for channel in range(3):
        samples_input = np.array([])
        samples_y = np.array([])
        samples_shap = np.array([])

        samples_input_x = np.array([])
        samples_input_y = np.array([])
        samples_y_x = np.array([])
        samples_y_y = np.array([])
        samples_shap_x = np.array([])
        samples_shap_y = np.array([])

        combined_df = pandas.DataFrame()
        # for sample_idx in tqdm(range(data_shap.shape[1])):
        for sample_idx in tqdm(range(1000)):

            sample_raw = data_x[sample_idx][channel][8:-8, 8:-8]
            # sample = sample_raw.reshape(-1)
            sample = (sample_raw - np.broadcast_to(np.mean(np.mean(sample_raw, axis=-1, keepdims=True), axis=-2, keepdims=True),
                                       sample_raw.shape)).reshape(-1)
            sample_loc_x, sample_loc_y = np.where(sample_raw)

            sample_y_raw = data_y[sample_idx][uvw]
            sample_y = sample_y_raw.reshape(-1)
            sample_y_x, sample_y_y = np.where(sample_y_raw)

            sample_shap_raw = data_shap[uvw][sample_idx][channel][8:-8, 8:-8]
            if NORMALIZE_SHAP:
                # full example in input_density_plots_just_tau_p.py -> over all channels at once
                shap_mean_value_over_channels = np.mean(data_shap[uvw][sample_idx][:, 8:-8, 8:-8])
                sample_shap_raw = (sample_shap_raw) / shap_mean_value_over_channels

            sample_shap = sample_shap_raw.reshape(-1)
            sample_shap_x, sample_shap_y = np.where(sample_shap_raw)

            samples_input = np.concatenate([samples_input, sample])
            samples_y = np.concatenate([samples_y, sample_y])
            samples_shap = np.concatenate([samples_shap, sample_shap])

            if REGRESSION:
                samples_input_x = np.concatenate([samples_input_x, sample_loc_x])
                samples_input_y = np.concatenate([samples_input_y, sample_loc_y])
                samples_y_x = np.concatenate([samples_y_x, sample_y_x])
                samples_y_y = np.concatenate([samples_y_y, sample_y_y])
                samples_shap_x = np.concatenate([samples_shap_x, sample_shap_x])
                samples_shap_y = np.concatenate([samples_shap_y, sample_shap_y])

            # create_contour_plot(sample, sample_shap)
            k = 0

        # clustering(samples_input, samples_shap)
        # colors = ['red', 'blue', 'green']
        colors = ['#440154', '#31688e', '#a0da39']
        line_types = ['-', '--', ':']
        top_x, top_y = create_contour_plot(samples_input, np.abs(samples_shap), uvw=uvw, channel=channel, name='input', return_x_y=True)
        tops.append([top_x, top_y])

        # axs.plot(top_x, top_y, color=colors[uvw], linestyle=line_types[channel], linewidth=2, label=f"$\phi({direction_names[uvw]}, {input_names[channel]})$")
        # plt.legend()
        # plt.ylabel('$\phi$')
        # plt.xlabel('Input')
        # plt.legend(fontsize=18)
        # plt.ylabel('$\phi$', fontsize=24)
        # plt.xlabel('Input', fontsize=24)
        # axs.tick_params(labelsize=20)


        # plt.savefig(f'./images/all_in_one.png',
        #             dpi=300, bbox_inches='tight')
        # create_contour_plot(samples_y, samples_shap/samples_y, uvw=uvw, channel=channel, name='output')
        # create_contour_plot(samples_y, np.abs(samples_shap), uvw=uvw, channel=channel, name='output')
        k = 0

        if REGRESSION:
            above_midline_x = 1*(samples_input_x > 96)
            above_midline_y = 1 * (samples_input_y > 96)


            samples_input_x_midlined = (192 - samples_input_x) * above_midline_x + samples_input_x * (1 - above_midline_x)
            samples_input_y_midlined = (192 - samples_input_y) * above_midline_y + samples_input_y * (1 - above_midline_y)
            # linear_fit(X=np.column_stack([samples_input, np.abs(samples_input), samples_y, np.abs(samples_y)]), y=np.abs(samples_shap), channel=channel, uvw=uvw)
            # directions = [np.abs(samples_input), np.abs(samples_y), samples_input_x, samples_input_y, samples_input_x_midlined, samples_input_y_midlined]
            directions = [np.abs(samples_input), np.abs(samples_y), samples_input_x_midlined, samples_input_y_midlined]
            if channel == 0 and uvw == 0:

                combined_df_started = linear_fit(X=np.column_stack(directions),
                           y=np.abs(samples_shap), channel=channel, uvw=uvw, combined_df=combined_df)
            else:
                combined_df_started = linear_fit(X=np.column_stack(directions),
                           y=np.abs(samples_shap), channel=channel, uvw=uvw, combined_df=combined_df_started)


        k = 0

# combined_df_started.to_excel(rf'regression_yp_{yp}_abs_top20_single_pixel.xlsx', index=False, header=False)

k=0

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


fig, axs = plt.subplots()
i = 0
for uvw in range(3):
    for channel in range(3):
        x, y = tops[i]
        # axs.plot(x, y, color=colors[uvw], linestyle=line_types[channel], linewidth=1.2,
        #          label=f"$\phi{direction_names[uvw]}, {input_names[channel]})$")
        axs.plot(x, y, color=colors[uvw], linestyle=line_types[channel], linewidth=1.2,
                 label=fr"$\widehat{{\phi}}_{{{direction_names[uvw]}, {input_names[channel]}}}$")

        # ax.set_title(fr'$\overline{{|\phi_{{{direction_names[j - 1][1:-1]}, {input_names[i][1:-1]}}}|}}$',
        #              bbox=dict(pad=0, facecolor='none', edgecolor='none'))

        i += 1
plt.legend(fontsize=7, bbox_to_anchor=(1.05, 0.5), loc='center left')
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.get_major_formatter().set_scientific(True)
ax.yaxis.get_major_formatter().set_powerlimits((0, 0))

ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
plt.locator_params(axis='y', nbins=4)
plt.locator_params(axis='x', nbins=7)

# top 0.940;    bottom 0.184;    left 0.138 (0.153 wuth widetilde);    right 0.704
# plt.xlabel(r'$\frac{{[\tau_{wx}, \tau_{wz}, p_{w}]_i - \text{mean}_i}}{std_i}$')
plt.xlabel(r'$[\widetilde{\tau}_{wx}, \widetilde{\tau}_{wz}, \widetilde{p}_{w}]$')
plt.ylabel(r'$\phi$')
if NORMALIZE_SHAP:
    plt.ylabel(r'$\widehat{\phi}$')
plt.tight_layout()
# plt.savefig(f'all_in_one_v4.png', dpi=300, bbox_inches='tight')
k = 0