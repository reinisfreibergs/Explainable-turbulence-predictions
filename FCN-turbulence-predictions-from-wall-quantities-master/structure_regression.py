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


yp = 15
data_shap = np.load(rf'final_shap_{yp}.npy')
data_x = np.load(rf'./x_test_{yp}.npy')
data_y = np.load(rf'./y_test_{yp}.npy')

# df_init = pandas.read_csv(f"structure_stats_abs_inputs_yp{yp}.csv")
# df_init = pandas.read_csv(f"structure_stats_abs_inputs_yp{yp}_top20.csv")
# df_init = pandas.read_csv(f"structure_stats_abs_inputs_yp{yp}_top10_min10.csv")
# df_init = pandas.read_csv(f"structure_stats_raw_inputs_yp{yp}_top10_min10.csv")
# df_init = pandas.read_csv(f"structure_stats_raw_inputs_yp{yp}_top20_min20.csv")

# df_init = pandas.read_csv(f"structure_stats_raw_inputs_yp{yp}_top10_min10.csv")

# df_init = pandas.read_csv(f"structure_stats_abs_inputs_yp15_top20_with_outputs_ranked_by_shap.csv")
# df_init = pandas.read_csv(f"structure_stats_abs_inputs_yp15_top20_with_outputs_ranked_by_input.csv")

df_init = pandas.read_csv(f"structure_stats_abs_inputs_yp15_top10_with_outputs_ranked_by_shap_step.csv")

values = df_init[(df_init['uvw'] == 0) & (df_init['channel'] == 2) & (df_init['area'] > 2)]
# # sort values by area key

# # plt.plot(values['area'], values['shap_abs_sum'])
# # add values['shap_abs_sum']/values['area'] as the last column
df_init['shap_abs_sum_per_area'] = df_init['shap_abs_sum']/df_init['area']
df_init['original_sum_per_area'] = df_init['original_sum']/df_init['area']
df_init['original_abs_sum_per_area'] = df_init['original_abs_sum']/df_init['area']
df_init['output_sum_per_area'] = df_init['output_sum']/df_init['area']
df_init['output_abs_sum_per_area'] = df_init['output_abs_sum']/df_init['area']

values['shap_abs_sum_per_area'] = values['shap_abs_sum']/values['area']
values = values.sort_values(by='shap_abs_sum_per_area')

# abs(df['original_abs_sum'])/df['original_abs_sum']
# sum every consecutive 10 values
# values['shap_abs_sum_per_area'] = values['shap_abs_sum_per_area'].rolling(10).sum()

def create_contour_plot(X, Y, show_mean=False, save=True, xname=None):
    # Create grid points for KDE
    xmin, xmax = 0.99*X.min(), 1.01*X.max()
    ymin, ymax = 0.99*Y.min(), 1.01*Y.max()
    bins = (100, 100)  # Number of bins for the histogram
    H_init, xedges, yedges = np.histogram2d(X, Y, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
    H = scipy.ndimage.gaussian_filter(H_init, sigma=2)

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    X_grid, Y_grid = np.meshgrid(xcenters, ycenters)

    num_levels = 10
    min_density = H[H > 0].min()  # Minimum non-zero density
    max_density = H.max()  # Maximum density
    levels = np.linspace(min_density, max_density, num_levels)

    # Plot the 2D histogram
    plt.figure(figsize=(22, 10))
    plt.imshow(H.T, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='viridis', aspect='auto')
    # plt.colorbar(label='Count')
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.title('2D Histogram Density Plot')
    # plt.scatter(X, Y)
    contour = plt.contour(X_grid, Y_grid, H.T, levels=levels, colors='red', linewidths=0.2, extent=[xmin, xmax, ymin, ymax])
    contourf = plt.contourf(X_grid, Y_grid, H.T, levels=levels, extent=[xmin, xmax, ymin, ymax])
    # plt.clabel(contour, inline=True, fontsize=8, colors='white')
    # [0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Add color bar
    # plt.colorbar(label='Count')

    plt.xlabel('log(area), grid points', fontsize=24)
    if xname:
        plt.xlabel(f'log({xname})', fontsize=32)
    if not show_mean:
        plt.ylabel(r'$log(\sum |\phi| + S)$', fontsize=24)
    else:
        plt.ylabel(r'$log(\frac{1}{N} \sum |\phi| + S)$', fontsize=24)
    # plt.title(f'Distribution of {input_names[channel]} input structure importance scores over {direction_names[uvw]} fluctuation', fontsize=24)
    plt.title(f'Distribution of {input_names[channel]} input structure {xname} importance scores over {direction_names[uvw]} fluctuation', fontsize=24)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.tight_layout()

    if xname is None:
        if save:
            if show_mean:
                plt.savefig(rf'./images_dist_v2/yp15_channel_{channel}_uvw_{uvw}_density_mean.png')
            else:
                plt.savefig(rf'./images_dist_v2/yp15_channel_{channel}_uvw_{uvw}_density.png')
        else:
            plt.show()

    else:
        if save:
            plt.savefig(rf'./images/structure_{xname}_yp15_channel_{channel}_uvw_{uvw}_density.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()

    # plt.show()
    # plt.close()

def create_distribution_plot(X, is_log=False):
    # save log and regular distribution plot for the structure areas
    # make the histogram of area distribution

    if is_log:
        X_to_plot = np.log(X)
    else:
        X_to_plot = X

    plt.hist(X_to_plot, bins=60, color='blue', alpha=0.7, figure=plt.figure(figsize=(22, 10)))


    plt.ylabel('Count', fontsize=24)
    plt.title(f'Area distribution for {input_names[channel]} input structures', fontsize=24)
    plt.tight_layout()
    # injcrease the font size foor the ticks
    # plt.xticks([np.min(X_to_plot)] + list(plt.xticks()[0]))

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    if is_log:
        plt.xlabel('log(area), grid points', fontsize=24)
        plt.savefig(rf'./images/yp15_channel_{channel}_uvw_{uvw}_log_structure_counts.png')
    else:
        plt.xlabel('area, grid points', fontsize=24)
        plt.savefig(rf'./images/yp15_channel_{channel}_uvw_{uvw}_structure_counts.png')
    plt.close()

def plot_spikes(area_sorted):

    plt.plot(area_sorted['area'], area_sorted['shap_abs_sum'], figure=plt.figure(figsize=(22, 10)))
    # add axis names
    plt.xlabel('area', fontsize=24)
    plt.ylabel('shap_abs_sum', fontsize=24)
    plt.title(f'shap_abs_sum vs area for {input_names[channel]} over {direction_names[uvw]} fluctuation', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    # plt.show()
    plt.close()
def save_area_plots(uvw, channel):
    area_sorted = df_init[(df_init['uvw'] == uvw) & (df_init['channel'] == channel) & (df_init['area'] > 9)].sort_values(by='area')

    # old 2d version with spikes
    # plot_spikes(area_sorted)

    # distribution plots
    # create_distribution_plot(area_sorted['area'], is_log=False)
    # create_distribution_plot(area_sorted['area'], is_log=True)

    # final density plots:
    X = np.array(area_sorted['area'])
    # X = np.array(area_sorted['length'])
    Y_raw = np.array(area_sorted['shap_abs_sum'])
    Y = np.array(area_sorted['shap_abs_sum'])/np.array(area_sorted['area'])


    # create_contour_plot(np.log(X), np.log(Y + 0.0000005), show_mean=True, save=True)
    # create_contour_plot(np.log(X), np.log(Y_raw + 0.0000005), show_mean=False, save=True)

    create_contour_plot(np.log(X), np.log(Y + 0.0000005), show_mean=True, save=False)
    create_contour_plot(np.log(X), np.log(Y_raw + 0.0000005), show_mean=False, save=False)



    # plt.savefig(rf'./images/area_shap_abs_sum_yp15_channel_{channel}_uvw_{uvw}.png')
    plt.close()


def save_structure_plots(uvw, channel):
    area_sorted = df_init[(df_init['uvw'] == uvw) & (df_init['channel'] == channel) & (df_init['area'] > 9)].sort_values(by='area')

    columns_to_avoid = ['Unnamed: 0', 'sample_idx', 'channel', 'uvw', 'original_sum']
    columns_to_plot = [col for col in area_sorted.columns if col not in columns_to_avoid]

    for column in columns_to_plot:
        X = np.array(area_sorted[column])
        if not np.sum(X<=0):
        if not np.isnan(np.log(X)).any() or not np.isinf(np.log(X)).any() and np.all(np.isfinite(np.log(X))):
                Y = np.array(area_sorted['shap_abs_sum']) / np.array(area_sorted['area'])
                create_contour_plot(np.log(X), np.log(Y + 0.0000005), show_mean=True, save=True, xname=column)
                # plt.savefig(rf'./images_dist_v2/yp15_channel_{channel}_uvw_{uvw}_density_mean.png')
                plt.close()


    k = 0
    # plt.savefig(rf'./images/area_shap_abs_sum_yp15_channel_{channel}_uvw_{uvw}.png')
    plt.close()

input_names = [r'$\tau_{wx}$ modified input', r'$\tau_{wz}$ modified input', r'$p_{w}$ modified input']
input_names = [r'$\tau_{wx}$', r'$\tau_{wz}$', r'$p_{w}$']
direction_names = ['u', 'v', 'w']
for uvw in range(3):
    for channel in tqdm(range(3)):
        save_area_plots(uvw, channel)
        # save_structure_plots(uvw, channel)

exit()


def create_simple_density_plot(X, Y):
    xmin, xmax = 0.99*X.min(), 1.01*X.max()
    ymin, ymax = 0.99*Y.min(), 1.01*Y.max()
    bins = (1000, 1000)  # Number of bins for the histogram
    H_init, xedges, yedges = np.histogram2d(X, Y, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
    H = scipy.ndimage.gaussian_filter(H_init, sigma=2)

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    X_grid, Y_grid = np.meshgrid(xcenters, ycenters)

    min_density = 0.01
    max_density = H.max()  # Maximum density

    # Plot the 2D histogram
    plt.figure(figsize=(16, 10))
    ax = plt.gca()
    im = plt.imshow(np.log10(H_init.T), origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='viridis', aspect='auto')
    contourf = plt.contourf(X_grid, Y_grid, np.log10(H.T), levels=25, extent=[xmin, xmax, ymin, ymax])
    plt.show()

combined_df = pandas.DataFrame()
for channel in range(3):
    for idx, uvw in enumerate(range(3)):

        # select only those samples where uvw = 0 and channel=0
        df = df_init[(df_init['uvw'] == uvw) & (df_init['channel'] == channel) & (df_init['area'] > 10)]
        #
        # X = df[['area', 'length', 'height', 'original_abs_sum_per_area', 'original_sum_per_area', 'mean_minimum_dist_to_wall', 'output_abs_sum_per_area', 'output_sum_per_area']].to_numpy()
        X = df[['length', 'height', 'original_abs_sum', 'original_sum', 'mean_minimum_dist_to_wall', 'output_abs_sum', 'output_sum']].to_numpy()

        # check if scaling works - it shouldnt matter since its a linear transformation
        # X[:, 0] *= 100

        # X = df[['area']].to_numpy()
        # y = df['shap_abs_sum'].to_numpy()
        # y = df['shap_abs_sum_per_area'].to_numpy()
        y = (df['shap_abs_sum']/df['area']).to_numpy()
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

# combined_df.to_excel(rf'regression_yp_{yp}_raw_top10_min10_scaled_per_area.xlsx', index=False, header=False)
k = 9




