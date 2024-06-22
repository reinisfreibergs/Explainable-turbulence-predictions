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



def create_contour_plot(X, Y, show_mean=False, save=False, uvw=0, channel=0, name='output'):
    input_names = [r'\tau_{wx}', r'\tau_{wz}', r'p_{w}']
    direction_names = ['u', 'v', 'w']
    # Create grid points for KDE
    xmin, xmax = 0.99*X.min(), 1.01*X.max()
    ymin, ymax = 0.99*Y.min(), 1.01*Y.max()
    bins = (1000, 1000)  # Number of bins for the histogram
    H_init, xedges, yedges = np.histogram2d(X, Y, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
    H = scipy.ndimage.gaussian_filter(H_init, sigma=2)

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    X_grid, Y_grid = np.meshgrid(xcenters, ycenters)

    num_levels = 1000
    # min_density = H[H > 0].min()  # Minimum non-zero density
    min_density = 0.01
    max_density = H.max()  # Maximum density
    # max_density = 0.01*H.max()
    levels = np.linspace(min_density, max_density, num_levels)



    # Plot the 2D histogram
    plt.figure(figsize=(16, 10))

    ax = plt.gca()
    # Set the y-axis to use scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.get_offset_text().set_fontsize(18)

    im = plt.imshow(np.log10(H_init.T), origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='viridis', aspect='auto')
    # plt.colorbar(label='Count')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.title('2D Histogram Density Plot')
    # plt.scatter(X, Y)
    # contour = plt.contour(X_grid, Y_grid, H.T, levels=levels, colors='red', linewidths=0.2, extent=[xmin, xmax, ymin, ymax], norm=LogNorm())
    # contourf = plt.contourf(X_grid, Y_grid, H.T, levels=levels, extent=[xmin, xmax, ymin, ymax], norm=LogNorm())
    # contour = plt.contour(X_grid, Y_grid, np.log(H.T), levels=20, colors='red', linewidths=0.2, extent=[xmin, xmax, ymin, ymax])
    contourf = plt.contourf(X_grid, Y_grid, np.log10(H.T), levels=25, extent=[xmin, xmax, ymin, ymax])
    cbar = plt.colorbar(im)
    cbar.ax.set_title('point count \n 10^', fontsize=20)
    cbar.ax.tick_params(labelsize=18)
    # plt.clabel(contour, inline=True, fontsize=8, colors='white')
    # [0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Add color bar
    # plt.colorbar(label='Count')

    if name == 'input':
        plt.xlabel(rf'${input_names[channel]}$', fontsize=24)
        # plt.title(
        #     f'${input_names[channel]}$ input SHAP value distribution for the ${direction_names[uvw]}$ fluctuation \n',
        #     fontsize=20)
    else:
        plt.xlabel(rf'${direction_names[uvw]}$', fontsize=24)
        # plt.title(
        #     f'${input_names[channel]}$ input SHAP value distribution for the ${direction_names[uvw]}$ fluctuation \n',
        #     fontsize=20)

    plt.ylabel(rf'$\phi({direction_names[uvw]}, {input_names[channel]})$', fontsize=24)
    plt.tick_params(labelsize=18)

    plt.tight_layout()
    plt.savefig(f'./images/{name}_abs_shap_scaled_log_density_{yp}_channel_{channel}_uvw_{uvw}.png',
                dpi=1200, bbox_inches='tight')
    # plt.show()

    k = 0
    plt.close()
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

def linear_fit(df_init):
    combined_df = pandas.DataFrame()
    for channel in range(3):
        for idx, uvw in enumerate(range(3)):

            # select only those samples where uvw = 0 and channel=0
            df = df_init[(df_init['uvw'] == uvw) & (df_init['channel'] == channel) & (df_init['area'] > 10)]
            #
            # X = df[['area', 'length', 'height', 'original_abs_sum_per_area', 'original_sum_per_area', 'mean_minimum_dist_to_wall', 'output_abs_sum_per_area', 'output_sum_per_area']].to_numpy()
            X = df[
                ['length', 'height', 'original_abs_sum', 'original_sum', 'mean_minimum_dist_to_wall', 'output_abs_sum',
                 'output_sum']].to_numpy()

            # check if scaling works - it shouldnt matter since its a linear transformation
            # X[:, 0] *= 100

            # X = df[['area']].to_numpy()
            # y = df['shap_abs_sum'].to_numpy()
            # y = df['shap_abs_sum_per_area'].to_numpy()
            y = (df['shap_abs_sum'] / df['area']).to_numpy()
            # X = np.concatenate()
            # add another column of ones to X

            scale = False
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

            if idx == 0:
                coefs_all = coefs
                r_squared_all = r_squared
            else:
                coefs_all = np.vstack((coefs_all, coefs))
                r_squared_all = np.vstack((r_squared_all, r_squared))
            k = 0

        df = pandas.DataFrame(np.hstack((r_squared_all, coefs_all)))
        # spacer_df = pandas.DataFrame([f'uvw: {uvw} channel: {channel}'])
        # iteration_df = pandas.concat([df, spacer_df], ignore_index=True)

        combined_df = pandas.concat([combined_df, df], ignore_index=True)



yp = 15
data_shap = np.load(rf'final_shap_{yp}.npy')
data_x = np.load(rf'./x_test_{yp}.npy')
data_y = np.load(rf'./y_test_{yp}.npy')


input_names = [r'$\tau_{wx}$', r'$\tau_{wz}$', r'$p_{w}$']
direction_names = ['u', 'v', 'w']
for uvw in range(3):
    for channel in range(3):
        samples_input = np.array([])
        samples_y = np.array([])
        samples_shap = np.array([])

        # for sample_idx in tqdm(range(len(data_x))):
        for sample_idx in tqdm(range(10)):

            sample = data_x[sample_idx][channel][8:-8, 8:-8].reshape(-1)
            sample_y = data_y[sample_idx][uvw].reshape(-1)
            sample_shap = data_shap[uvw][sample_idx][channel][8:-8, 8:-8].reshape(-1)

            samples_input = np.concatenate([samples_input, sample])
            samples_y = np.concatenate([samples_y, sample_y])
            samples_shap = np.concatenate([samples_shap, sample_shap])

            # create_contour_plot(sample, sample_shap)
            k = 0



        create_contour_plot(samples_input, np.abs(samples_shap), uvw=uvw, channel=channel, name='input')
        # create_contour_plot(samples_y, samples_shap/samples_y, uvw=uvw, channel=channel, name='output')
        create_contour_plot(samples_y, np.abs(samples_shap), uvw=uvw, channel=channel, name='output')

        # scaler = preprocessing.StandardScaler()
        # X = scaler.fit_transform(X)
        k = 0


k = 0