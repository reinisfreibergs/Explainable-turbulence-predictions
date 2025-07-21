
import numpy as np
import scipy
from scipy.signal import savgol_filter
import pickle
import argparse


def create_contour_plot(X, Y, show_mean=False, save=False, uvw=0, channel=0, name='output', return_x_y = False):

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


        return top_1_percent_x[minx:maxx], top_1_percent_y_smooth[minx:maxx]

    k = 0

parser = argparse.ArgumentParser(description="Run data processing with specified 'yp' and 'ret' values.")
parser.add_argument('--yp', type=int, default=50)
parser.add_argument('--ret', type=int, default=180)
args = parser.parse_args()

Ret = args.ret
yp = args.yp
# data_shap = np.load(rf'final_shap_{yp}_{Ret}_100.npy')
# data_x = np.load(rf'./data_x_{yp}_{Ret}_100.npy')
# data_y = np.load(rf'./data_y_{yp}_{Ret}_100.npy')
if Ret == 550:
    data_shap = np.load(rf'final_shap_{yp}_{Ret}.npy').squeeze()
    data_x = np.load(rf'./data_x_{yp}_{Ret}.npy').squeeze()
    data_y = np.load(rf'./data_y_{yp}_{Ret}.npy').squeeze()
else:
    data_shap = np.load(rf'final_shap_{yp}.npy').squeeze()
    data_x = np.load(rf'./x_test_{yp}.npy').squeeze()
    data_y = np.load(rf'./y_test_{yp}.npy').squeeze()


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

            shap_mean_values_per_sample = np.mean(np.abs(shap_for_norm), axis=(1, 2, 3))

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

# pickle save the tops

with open(f'tops_{yp}_{Ret}.pkl', 'wb') as f:
    pickle.dump(tops, f)