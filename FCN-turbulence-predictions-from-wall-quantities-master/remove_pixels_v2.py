#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import time
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import platform
if platform.system() == 'Windows':
    import matplotlib
    matplotlib.use("qt5agg")

from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import shap
from tqdm import tqdm
from skimage.transform import resize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, '../conf')
sys.path.insert(0, '../models')

# Training utils
from src.training_utils import load_trained_model
from src.tfrecord_utils import get_dataset
# from conf.config_sample import WallRecon

import config

prb_def = 'WallRecon'


app = config.WallRecon

os.environ["CUDA_VISIBLE_DEVICES"] = str(app.WHICH_GPU_TEST);
# os.environ["CUDA_VISIBLE_DEVICES"]="";

# =============================================================================
#   IMPLEMENTATION WARNINGS
# =============================================================================

# Data augmentation not implemented in this model for now
app.DATA_AUG = False
# Transfer learning not implemented in this model for now
app.TRANSFER_LEARNING = False

# %% Hardware detection and parallelization strategy
physical_devices = tf.config.list_physical_devices('GPU')
availale_GPUs = len(physical_devices)
print('Using TensorFlow version:', tf.__version__, ', GPU:', availale_GPUs)
# print(tf.keras.__version__)

if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

on_GPU = app.ON_GPU
n_gpus = app.N_GPU

distributed_training = on_GPU == True and n_gpus > 1

# %% Dataset and ANN model

tstamp = app.TIMESTAMP

dataset_test, X_test, n_samples_tot, model_config = \
    get_dataset(prb_def, app, timestamp=tstamp,
                train=False, distributed_training=distributed_training)


ds_path = app.DS_PATH_TEST
# Average profiles folder
avg_path = ds_path + '/.avg/'
avgs = tf.reshape(tf.constant(np.loadtxt(avg_path + 'mean_' +
                                         app.VARS_NAME_OUT[0] + '.m').astype(np.float32)[:, 1]), (1, -1))
for i in range(1, app.N_VARS_OUT):
    avgs = tf.concat((avgs, tf.reshape(tf.constant(
        np.loadtxt(avg_path + 'mean_' +
                   app.VARS_NAME_OUT[i] + '.m').astype(np.float32)[:, 1]), (1, -1))), 0)

print('')
print('# ====================================================================')
print('#     Summary of the options for the model                            ')
print('# ====================================================================')
print('')
print(f'Number of samples for training: {int(n_samples_tot)}')
# print(f'Number of samples for validation: {int(n_samp_valid)}')
print(f'Total number of samples: {n_samples_tot}')
print(f"Batch size: {model_config['batch_size']}")
print('')
print(f'Data augmentation: {app.DATA_AUG} (not implemented in this model)')
print(f'Initial distribution of parameters: {app.INIT}')
if app.INIT == 'random':
    print('')
    print('')
if app.INIT == 'model':
    print(f'    Timestamp: {app.INIT_MODEL[-10]}')
    print(f'    Transfer learning: {app.TRANSFER_LEARNING} (not implemented in this model)')
print(f'Prediction of fluctuation only: {app.FLUCTUATIONS_PRED}')
print(f'y- and z-output scaling with the ratio of RMS values : {app.SCALE_OUTPUT}')
print(f'Normalized input: {app.NORMALIZE_INPUT}')
# print(f'File for input statistics: {app.AVG_INPUTS_FILE}')
print('')
print('# ====================================================================')
# %% Testing

# Iterating over ground truth datasets ----------------------------------------
itr = iter(dataset_test)
itrX = iter(X_test)

X_test = np.ndarray((n_samples_tot, app.N_VARS_IN,
                     model_config['nx_'] + model_config['pad'],
                     model_config['nz_'] + model_config['pad']),
                    dtype='float')
Y_test = np.ndarray((n_samples_tot, app.N_VARS_OUT,
                     model_config['nx_'],
                     model_config['nz_']),
                    dtype='float')
ii = 0
for i in range(n_samples_tot):
    X_test[i] = next(itrX)
    if app.N_VARS_OUT == 1:
        Y_test[i, 0] = next(itr)
    elif app.N_VARS_OUT == 2:
        (Y_test[i, 0], Y_test[i, 1]) = next(itr)
    else:
        (Y_test[i, 0], Y_test[i, 1], Y_test[i, 2]) = next(itr)
    ii += 1
    # print(i+1)
print(f'Iterated over {ii} samples')
print('')

# Preparation for saving the results ------------------------------------------
pred_path = app.CUR_PATH + '/.predictions/'
if not os.path.exists(pred_path):
    os.mkdir(pred_path)

# pred_path = cur_path+'/CNN-'+timestamp+'-ckpt/'
pred_path = pred_path + model_config['name'] + '/'
if not os.path.exists(pred_path):
    os.mkdir(pred_path)

model_config['pred_path'] = pred_path

# Loading the neural network --------------------------------------------------
CNN_model = load_trained_model(model_config)

# Testing
Y_pred = np.ndarray((n_samples_tot, app.N_VARS_OUT,
                     model_config['nx_'], model_config['nz_']), dtype='float')

def plot_two_img_for_comparison(img1, img2, comparative_idx=0, idx_to_plot=0):
    if len(img1) == 3:
        img1 = img1[None, :]
    if len(img2) == 3:
        img2 = img2[None, :]
    # idx_to_plot = 0
    fig, axs = plt.subplots(1, 2)
    sample1 = img1[idx_to_plot][comparative_idx]
    sample2 = img2[idx_to_plot][comparative_idx]

    axs[0].imshow(img1[idx_to_plot][comparative_idx])
    axs[0].set_title(f'truth, max:{round(np.max(sample1), 2)}, min:{round(np.min(sample1), 2)}', fontsize=20)
    axs[1].imshow(img2[idx_to_plot][comparative_idx])
    axs[1].set_title(f'predicted, max:{round(np.max(sample2), 2)}, min:{round(np.min(sample2), 2)}', fontsize=20)
    plt.suptitle(f'yp:{app.TARGET_YP}, sample number: {idx_to_plot}', fontsize=20)



def create_final_plot_single_pixel(result_row):
    index_u_v_w, first_np, second_total, mse_total, true_value, ranked_indices, sample_idx, pixel_frequency, idxes = result_row

    # u as 0 since that was the first selected -> finish with 9 plots
    # index_u_v_w = 0

    plt.figure(figsize=(22, 10))
    reference_mse = np.mean((true_value[index_u_v_w] - first_np[0, index_u_v_w]) ** 2)
    # ranked_indices_idxes = list(range(0, len(ranked_indices), pixel_frequency))
    input_names = [r'$\tau_{wx}$ modified input', r'$\tau_{wz}$ modified input', r'$p_{w}$ modified input']
    direction_names = ['u', 'v', 'w']
    plt.hlines(y=0, xmin=0, xmax=max(idxes), colors='gray', linestyles='--')
    for channel_idx in range(3):
        plt.plot(idxes, np.array((mse_total[channel_idx][:, index_u_v_w] - reference_mse)/reference_mse*100), label=input_names[channel_idx])
    plt.legend(fontsize=24)
    plt.title(rf'Change in MSE of the ${direction_names[index_u_v_w]}$ prediction after removing a single pixel ordered by absolute mean SHAP, sample idx: {sample_idx}', fontsize=24)
    plt.xlabel(rf'Fraction of pixels removed from input', fontsize=24)
    plt.ylabel(rf'Relative increase in MSE, %', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(rf'./images/remove_pixels_by_fraction_abs_specific_yp{app.TARGET_YP}_{index_u_v_w}_sample_{sample_idx}_max_{np.max(fractions)}.png')
    # plt.show()


def create_final_plot_continous_pixels_fixed_full_reference_mse(result_row):
    index_u_v_w, first_np, second_total, mse_total, true_value, ranked_indices, sample_idx, pixel_frequency, idxes = result_row

    # u as 0 since that was the first selected -> finish with 9 plots
    # index_u_v_w = 0

    plt.figure(figsize=(18, 10))
    reference_mse = np.mean((true_value - first_np.squeeze()) ** 2)
    # ranked_indices_idxes = list(range(0, len(ranked_indices), pixel_frequency))
    input_names = [r'$\tau_{wx}$ modified input', r'$\tau_{wz}$ modified input', r'$p_{w}$ modified input']
    direction_names = ['u', 'v', 'w']
    plt.hlines(y=0, xmin=0, xmax=max(idxes), colors='gray', linestyles='--')
    for channel_idx in range(3):
        # plt.plot(idxes, np.array((mse_total[channel_idx][:, index_u_v_w] - reference_mse)/reference_mse*100), label=input_names[channel_idx])
        modified_image_mse = np.mean(mse_total[channel_idx], axis=-1)
        relative_increase = (modified_image_mse - reference_mse)/reference_mse * 100

        #  fix the coefs as true is (3,192, 192) but second_total is (3,10,3,192,192)

        increase_images = np.mean((true_value - np.array(second_total)[channel_idx]) ** 2, axis=(1))

        plt.plot(idxes, relative_increase, label=input_names[channel_idx], linewidth=2)

        # lets look at one image and add the max error annotation
        # max_pos = np.unravel_index(np.argmax(data), data.shape)
        # max_value = data[max_pos]
        #
        # # Create the plot
        # plt.imshow(data, cmap='viridis')
        # plt.colorbar()
        #
        # # Add a pointer to the maximum value
        # plt.annotate(f'Max: {max_value:.2f}', xy=max_pos, xytext=(max_pos[1] + 2, max_pos[0] - 2),
        #              arrowprops=dict(facecolor='white', shrink=0.05, width=2))

    plt.legend(fontsize=24)
    plt.title(rf'Change in MSE of the ${direction_names[index_u_v_w]}$ prediction after removing a fraction of pixels ordered by absolute SHAP, sample idx: {sample_idx}', fontsize=24)
    plt.xlabel(rf'Fraction of pixels removed from input', fontsize=24)
    # plt.ylabel(rf'Relative increase in MSE, %', fontsize=24)
    plt.title(r'Relative increase in MSE, %: $\frac{{\text{mse\_orig} - \text{mse\_removed}}}{{\text{mse\_orig}}} \times 100$', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(rf'./fractions_removed_averaged/images/remove_pixels_by_fraction_abs_specific_yp{app.TARGET_YP}_{index_u_v_w}_sample_{sample_idx}_max_{np.max(fractions)}.png')
    # plt.show()

def explain_the_huge_increase_in_mse(result_row):
    index_u_v_w, first_np, second_total, mse_total, true_value, ranked_indices, sample_idx, pixel_frequency, idxes = result_row

    # u as 0 since that was the first selected -> finish with 9 plots
    # index_u_v_w = 0

    plt.figure(figsize=(22, 10))
    # this reference mse describes only the error in that particular channel
    reference_mse = np.mean((true_value[index_u_v_w] - first_np[0, index_u_v_w]) ** 2)
    # ranked_indices_idxes = list(range(0, len(ranked_indices), pixel_frequency))
    reference_mse_full = np.mean((true_value - first_np.squeeze()) ** 2)

    for channel_idx in range(3):
        relative_increases = np.array((mse_total[channel_idx][:, index_u_v_w] - reference_mse)/reference_mse*100)
        relative_increases_img = true_value[index_u_v_w] - second_total[channel_idx][:, index_u_v_w]

        # most points removed image
        img_most_removed = second_total[channel_idx][-1]  # (3, 192, 192)


        np.mean((true_value - second_np) ** 2, axis=(2, 3))


data_shap = np.load(rf'final_shap_{app.TARGET_YP}.npy')
data_x = np.load(rf'./x_test_{app.TARGET_YP}.npy')
data_y = np.load(rf'./y_test_{app.TARGET_YP}.npy')
mean_shap = np.mean(np.abs(data_shap),axis=1)


u_rms = model_config['rms'][0][model_config['ypos_Ret'][str(app.TARGET_YP)]]
default_scale = np.array([[(model_config['rms'][i][model_config['ypos_Ret'][str(app.TARGET_YP)]] / u_rms).numpy(),
                           avgs[i][model_config['ypos_Ret'][str(app.TARGET_YP)]].numpy()] for i in range(3)])


def replace_fraction_of_pixels(orig_img, fraction, background_value, ranked_indices, channel):

    modified_img = np.array(orig_img)
    # it sorts starting from smallest so select from back
    for pix_idx in range(0, int(fraction*len(ranked_indices))):
        modified_img[channel][ranked_indices[-(pix_idx + 1)][0], ranked_indices[-(pix_idx + 1)][1]] = background_value

    return modified_img

def create_modified_prediction_continuous_pixels(sample_idx=0, pixel_frequency=100, index_u_v_w = 0, fractions = [0]):
    # v1 - remove the n-th most important pixel and track the change in mse.

    mse_total = []
    second_total = []
    true_total = []
    input_total = []
    true_value = np.array(Y_test[sample_idx])
    first = CNN_model.predict(data_x[sample_idx][None, :])
    for channel in range(3):

        ranked_indices = np.array(np.unravel_index(np.argsort(np.abs(data_shap)[index_u_v_w][sample_idx][channel].flatten()), data_shap[index_u_v_w][sample_idx][channel].shape)).T
        current_mean = np.mean(X_test[sample_idx], axis=(-1, -2))

        modified_img = np.array(X_test[sample_idx])
        # use every 100th for now as every single one will be too much
        for idx, fraction in enumerate(fractions):

            # here instead of replacing the single pixel replace all until reaching the fraction
            modified_img = replace_fraction_of_pixels(modified_img.squeeze(),
                                                      fraction=fraction,
                                                      background_value=current_mean[channel],
                                                      ranked_indices=ranked_indices,
                                                      channel=channel)


            modified_img = modified_img[None, :]
            if idx == 0:
                modified_img_total = modified_img
            else:
                modified_img_total = np.concatenate((modified_img_total, modified_img))
            # modified_img[channel][ranked_indices[pix_idx][0], ranked_indices[pix_idx][1]] = current_mean[0]


        second = CNN_model.predict(modified_img_total, batch_size=2)

        final_mean, final_std = default_scale[:, 1], default_scale[:, 0]
        first_np = final_std[None, :, None, None] * np.concatenate(first, axis=1)
        second_np = final_std[None, :, None, None] * np.concatenate(second, axis=1)

        # second_np -> (tries,  u,v,w,  )
        mse_values = np.mean((true_value - second_np) ** 2, axis=(2, 3))
        mse_total.append(mse_values)
        second_total.append(second_np)
        true_total.append(true_value)
        input_total.append(modified_img_total)
        # plt.plot(mse_values)
        # first_loss = np.mean((true_value - first_np)**2)
        # second_loss = np.mean((true_value - second_np)**2)

    return index_u_v_w, first_np, second_total, mse_total, true_value, ranked_indices, sample_idx, pixel_frequency, fractions



def create_single_comparison_plot(sample_idx=0, pixel_frequency=100, index_u_v_w = 0, fractions = [0]):
    # v1 - remove the n-th most important pixel and track the change in mse.

    mse_total = []
    second_total = []
    true_total = []
    input_total = []
    true_value = np.array(Y_test[sample_idx])
    first = CNN_model.predict(data_x[sample_idx][None, :])
    # use only the last pressure channel
    for channel in [2]:

        ranked_indices = np.array(np.unravel_index(np.argsort(np.abs(data_shap)[index_u_v_w][sample_idx][channel].flatten()), data_shap[index_u_v_w][sample_idx][channel].shape)).T
        current_mean = np.mean(X_test[sample_idx], axis=(-1, -2))

        modified_img = np.array(X_test[sample_idx])
        # use every 100th for now as every single one will be too much
        for idx, fraction in enumerate(fractions):

            # here instead of replacing the single pixel replace all until reaching the fraction
            modified_img = replace_fraction_of_pixels(modified_img.squeeze(),
                                                      fraction=fraction,
                                                      background_value=current_mean[channel],
                                                      ranked_indices=ranked_indices,
                                                      channel=channel)


            modified_img = modified_img[None, :]
            if idx == 0:
                modified_img_total = modified_img
            else:
                modified_img_total = np.concatenate((modified_img_total, modified_img))
            # modified_img[channel][ranked_indices[pix_idx][0], ranked_indices[pix_idx][1]] = current_mean[0]


        second = CNN_model.predict(modified_img_total, batch_size=2)


        fig = plt.figure(figsize=(24, 16))
        axs = ImageGrid(fig, 111,
                         nrows_ncols=(2, 2),
                         cbar_location="right",
                         cbar_mode='edge',
                         direction='row',
                         cbar_size="5%",
                         cbar_pad=0.2,
                         share_all=True,
                         axes_pad=0.4
                         )
        # fig, axs = plt.subplots(2, 2, figsize=(18, 10))

        vmin = np.min(resize(data_x[sample_idx][2, 8:-8, 8:-8], (192, 384)))
        vmax = np.max(resize(data_x[sample_idx][2, 8:-8, 8:-8], (192, 384)))

        vmin3 = np.min(resize(first[0][0][0], (192, 384)))
        vmax3 = np.max(resize(first[0][0][0], (192, 384)))

        im1 = axs[0].imshow(resize(data_x[sample_idx][2, 8:-8, 8:-8], (192, 384)), cmap='RdBu_r', vmin=vmin,
                               vmax=vmax, extent=[-2*np.pi, 2*np.pi, -np.pi, np.pi])

        im2 = axs[1].imshow(resize(modified_img_total.squeeze()[2, 8:-8, 8:-8], (192, 384)), cmap='RdBu_r',
                               vmin=vmin, vmax=vmax, extent=[-2*np.pi, 2*np.pi, -np.pi, np.pi])  # modified input

        im3 = axs[2].imshow(resize(first[0][0][0], (192, 384)), cmap='RdBu_r', vmin=vmin3, vmax=vmax3, extent=[-2*np.pi, 2*np.pi, -np.pi, np.pi])  # original prediction

        im4 = axs[3].imshow(resize(second[0][0][0], (192, 384)), cmap='RdBu_r', vmin=vmin3, vmax=vmax3, extent=[-2*np.pi, 2*np.pi, -np.pi, np.pi])

        axs[0].set_title('Original $p_w$ input', fontsize=24)
        axs[1].set_title('Modified $p_w$ input', fontsize=24)
        axs[2].set_title('Original $u$ prediction', fontsize=24)
        axs[3].set_title('Modified $u$ prediction', fontsize=24)

        axs[0].set_ylabel(r'$z/h$', fontsize=24)
        axs[2].set_ylabel(r'$z/h$', fontsize=24)
        axs[2].set_xlabel(r'$x/h$', fontsize=24)
        axs[3].set_xlabel(r'$x/h$', fontsize=24)

        # plt.subplots_adjust(wspace=0.3, hspace=0.3)
        cbar1 = plt.colorbar(im2, cax=axs.cbar_axes[0])
        cbar2 = plt.colorbar(im4, cax=axs.cbar_axes[1])
        cbar1.ax.tick_params(labelsize=18)
        cbar2.ax.tick_params(labelsize=18)

        custom_ticks = [vmin3, -0.2, -0.1, 0, 0.1, 0.2, vmax3]

        # Set the ticks and custom formatted labels
        cbar2.set_ticks(custom_ticks)
        cbar2.set_ticklabels(
            [f'min={vmin3:.2f}' if tick == vmin3 else (f'max={vmax3:.2f}' if tick == vmax3 else f'{tick:.2f}') for tick
             in custom_ticks])

        max_pos = np.unravel_index(np.argmax(resize(second[0][0][0], (192, 384))), resize(second[0][0][0], (192, 384)).shape)
        max_value = resize(second[0][0][0], (192, 384))[max_pos]
        extent = [-2 * np.pi, 2 * np.pi, -np.pi, np.pi]
        # Convert pixel coordinates to data coordinates
        x_max = extent[0] + (extent[1] - extent[0]) * max_pos[1] / 384
        y_max = extent[2] + (extent[3] - extent[2]) * (192 - max_pos[0]) / 192


        # Add a pointer to the maximum value
        axs[3].annotate(f'max pred: {max_value:.2f}', xy=(x_max, y_max), xytext=(x_max + 0.4, y_max + 0.4),
                        arrowprops=dict(facecolor='white', shrink=0.05, width=2),
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"), fontsize=17)

        for ax in axs:
            ax.tick_params(labelsize=18)

        # scatter the replaced values to be better visible
        y_ranked_indices = ranked_indices[-(int(0.01 * len(ranked_indices)) + 1):][:, 0] - 8
        x_ranked_indices = 2*ranked_indices[-(int(0.01 * len(ranked_indices)) + 1):][:, 1] - 16
        valid_points = (x_ranked_indices >= 0) * (x_ranked_indices < 384) * (y_ranked_indices >= 0) * (y_ranked_indices < 192)
        x_ranked_indices = x_ranked_indices[valid_points]
        y_ranked_indices = y_ranked_indices[valid_points]
        x_ranked_indices_scaled = extent[0] + (extent[1] - extent[0]) * x_ranked_indices / 384
        y_ranked_indices_scaled = extent[2] + (extent[3] - extent[2]) * (192 - y_ranked_indices) / 192

        axs[1].scatter(x_ranked_indices_scaled, y_ranked_indices_scaled, s=6, c='black', marker=',')


        # Calculate the position for the text label
        bbox = cbar2.ax.get_position()
        x_pos = bbox.x0 + bbox.width / 2
        y_pos = bbox.y1 + 0.01
        # Add a text label just above the lower colorbar
        fig.text(x_pos, y_pos, f'Values capped at original max', ha='center', va='bottom', rotation=0, fontsize=16)


        plt.savefig(f'./images/one_percent_removal_pressure_yp_{config.WallRecon.TARGET_YP}_top_{fractions[0]}_v2.png', dpi=1200, bbox_inches='tight')
        k = 0






        # now save the second side-by-side comparison plot with the absolute difference vs removed pixels
        difference_image = resize(second[index_u_v_w].squeeze(), (192, 384)) - resize(first[index_u_v_w].squeeze(), (192, 384))
        difference_image = difference_image * (1*np.abs(difference_image) > 0.0001)
        norm = TwoSlopeNorm(
            vmin=np.min(difference_image),
            vcenter=0,
            vmax=np.max(difference_image))
        fig_diff, ax_diff = plt.subplots(figsize=(12, 6))
        diff_img = ax_diff.imshow(difference_image, cmap='RdBu_r', extent=[-2 * np.pi, 2 * np.pi, -np.pi, np.pi], norm=norm)
        ax_diff.scatter(x_ranked_indices_scaled, y_ranked_indices_scaled, s=6, c='green', marker=',', alpha=0.4)
        # plt.colorbar(diff_img)


        divider = make_axes_locatable(ax_diff)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig_diff.colorbar(diff_img, cax=cax, spacing='proportional')
        cbar.ax.set_yscale('linear')
        cbar.ax.tick_params(labelsize=18)

        ticks = np.concatenate((np.linspace(np.min(difference_image), 0, num=2), np.linspace(0, np.max(difference_image), num=6)[1:]))
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{tick:.2f}' for tick in ticks])


        # Adjust the layout to match the height
        ax_diff.set_aspect(aspect='auto')
        ax_diff.tick_params(labelsize=18)
        ax_diff.set_ylabel(r'$z/h$', fontsize=24)
        ax_diff.set_xlabel(r'$x/h$', fontsize=24)
        ax_diff.set_title('Difference between $u$ prediction with modified and original $p_w$ input \n $u_{baseline_{p_w}} - u_{modified_{p_w}}$',
                          fontsize=22)

        plt.savefig(f'./images/one_percent_removal_pressure_yp_{config.WallRecon.TARGET_YP}_top_{fractions[0]}_v2_comparison_alpha04.png',
                    dpi=1200, bbox_inches='tight')



        # finally make side-by-side images with true-original and true-modified







    # return index_u_v_w, first_np, second_total, mse_total, true_value, ranked_indices, sample_idx, pixel_frequency, fractions

# for fractions in [list(np.linspace(0, 1, 10)), list(np.linspace(0, 0.1, 10))]:
#     for index_uvw in range(3):
#         result_row = create_modified_prediction_continuous_pixels(sample_idx=0, pixel_frequency=100, index_u_v_w=index_uvw, fractions=fractions)
#         # create_final_plot_single_pixel(result_row)
#         create_final_plot_continous_pixels_fixed_full_reference_mse(result_row)

# for frac in tqdm([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]):
for frac in tqdm([0.01]):
    create_single_comparison_plot(sample_idx=0, pixel_frequency=100, index_u_v_w=0, fractions=[frac])

k = 0







# first_np, second_np, model_img = create_modified_prediction()
# fig, axes = plt.subplots(nrows=3, ncols=2)
# for i in range(3):
#     ax = axes[i]
#     ax[0].imshow(first_np[i], cmap='coolwarm')
#     ax[1].imshow(second_np[i], cmap='coolwarm')
#
# axes[0][0].set_title('original prediction')
# axes[0][1].set_title(f'modified by removing top 1 pixel')