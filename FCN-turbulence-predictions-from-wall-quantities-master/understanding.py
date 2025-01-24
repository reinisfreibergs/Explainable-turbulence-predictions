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
import shutil


from tqdm import tqdm
from skimage.transform import resize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid

# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
plt.rcParams['figure.figsize'] = (3.5, 2.5)
plt.rcParams['figure.dpi'] = 300
# Font settings
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# Axes and labels
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['lines.linewidth'] = 1.0
# Legends and titles
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rc('text', usetex=True)

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


ITERATE = False
if ITERATE:
    for channel in range(3):
        ranks = []
        max_errors = []
        for file in os.listdir(rf'./fractions_removed'):
            if file[-3:] == 'npz':
                if file[-5] == str(channel):
                    print(file)
                    data = np.load(rf'./fractions_removed/{file}')
                    first = data['first']
                    second = data['second']
                    modified_img_total = data['modified_img_total']
                    idx_rank = data['idx_rank']
                    max_error = data['max_error']
                    # plot_two_img_for_comparison(first, second, comparative_idx=2, idx_to_plot=0)
                    ranks.append(idx_rank)
                    max_errors.append(max_error)

                    plt.imshow(np.abs(second[0][0][0] - first[0][0][0]))
                    # savefig at the channel folder
                    plt.savefig(rf'./fractions_removed/removal_error_pressure_yp_{app.TARGET_YP}_channel_{channel}_top_{file.split("_")[-1][:-4]}.png')
                    k = 0
        k = 0

# for channel in range(3):
#     images = []
#     for idx, fraction in tqdm(enumerate(np.linspace(0, 1, 100))):
#         if idx == 0:
#             fraction = '0.000'
#         image = rf'./fractions_removed/removal_error_pressure_yp_{app.TARGET_YP}_channel_{channel}_top_{fraction}.png'
#         # laod the image and append to the list
#         # images.append(plt.imread(image))
#         # copy the images by channel and uvw folders
#         if not os.path.exists(rf'./images_by_channel/{channel}'):
#             os.makedirs(rf'./images_by_channel/{channel}')
#         # use shutil copy
#         shutil.copy(image, rf'./images_by_channel/{channel}/removal_error_pressure_yp_{app.TARGET_YP}_channel_{channel}_top_{fraction}.png')
#     k = 0

# understanding plots

plt.rc('text', usetex=True)
plt.rcParams['font.size'] = 10 + 2
plt.rcParams['font.family'] = 'serif'
input_names = [r'$\tau_{wx}$', r'$\tau_{wz}$', r'$p_{w}$']
colors = ['#a0da39', '#440154', '#31688e']
fig, axs = plt.subplots()
fractions = np.linspace(0, 1, 100)
for channel in range(3):
    ranks = []
    max_errors = []
    for file in os.listdir(rf'./fractions_removed'):
        if file[-3:] == 'npz':
            if file[-5] == str(channel):
                print(file)
                data = np.load(rf'./fractions_removed/{file}')
                first = data['first']
                second = data['second']
                modified_img_total = data['modified_img_total']
                idx_rank = data['idx_rank']
                max_error = data['max_error']
                # plot_two_img_for_comparison(first, second, comparative_idx=2, idx_to_plot=0)
                ranks.append(idx_rank)
                max_errors.append(max_error)



                k = 0

    axs.plot(np.linspace(0, 1, len(ranks)), [i/(228**2) for i in ranks], label=input_names[channel], color=colors[channel])
    # axs.set_xlabel('Fraction of pixels removed from input')
    # axs.set_ylabel('Fraction of pixels contributing \n to 50\% of the total error')
plt.rcParams['text.usetex'] = True
# plt.rc('text', usetex=True)
plt.xlabel(r'$\frac{{\rm{px}}_{\rm{rmv}}}{{\rm{px}}_{\rm{tot}}}$', fontsize=14)
plt.ylabel(r'$\frac{{\rm{px}}_{\rm{rmv}}}{{\rm{px}}_{\rm{tot}}}\mid_{\rm{50\% error}}$', fontsize=14)
# plt.ylabel(r'$\frac{{\rm{px}}_{\rm{rmv}}}{{\rm{px}}_{\rm{tot}}} \right|_{\text{50\% error}}$')
# plt.xlabel(r'$\frac{\tau_{wx} - \overline{\tau_{wx}}}{\sigma_{\tau_{wx}}}$', fontsize=14)


    # axs.tick_params(labelsize=18)
    # plt.title('Fraction of pixels that contribute to 50% of the error relative to the total number of input pixels removed sorted by SHAP importance')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f'./volume_50perc_error_v2.png', dpi=300, bbox_inches='tight')
plt.close()


z = 0

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


data_shap = np.load(rf'final_shap_{app.TARGET_YP}.npy')
data_x = np.load(rf'./x_test_{app.TARGET_YP}.npy')
data_y = np.load(rf'./y_test_{app.TARGET_YP}.npy')
mean_shap = np.mean(np.abs(data_shap),axis=1)


u_rms = model_config['rms'][0][model_config['ypos_Ret'][str(app.TARGET_YP)]]
default_scale = np.array([[(model_config['rms'][i][model_config['ypos_Ret'][str(app.TARGET_YP)]] / u_rms).numpy(),
                           avgs[i][model_config['ypos_Ret'][str(app.TARGET_YP)]].numpy()] for i in range(3)])


def replace_fraction_of_pixels(orig_img, fraction, background_value, ranked_indices, channel):

    try:
        modified_img = np.array(orig_img)
        # it sorts starting from smallest so select from back
        for pix_idx in range(0, int(fraction*len(ranked_indices))):
            modified_img[channel][ranked_indices[-(pix_idx + 1)][0], ranked_indices[-(pix_idx + 1)][1]] = background_value
    except Exception as e:
        print(e)
        k = 0

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


        image4 = np.abs(second[0][0][0] - first[0][0][0])

        total_abs_error = np.sum(image4)
        ranked_indices_image_4 = np.array(np.unravel_index(np.argsort(image4.flatten()), image4.shape)).T

        # plot modified image and image 4 side by side with colorbars
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        im = axs[0].imshow(resize(modified_img_total.squeeze()[2, 15:-15, 15:-15], (178, 356)), extent=[-2*np.pi, 2*np.pi, -np.pi, np.pi], cmap='RdBu_r')
        axs[0].set_title('Modified input')
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

        im = axs[1].imshow(resize(image4, (192, 192*2)), extent=[-2*np.pi, 2*np.pi, -np.pi, np.pi], cmap='RdBu_r')
        axs[1].set_title('Absolute error')
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        plt.tight_layout()
        plt.savefig(f'./fractions_removed/removal_error_pressure_yp_{app.TARGET_YP}_channel_{channel}_top_{fractions[0]}.png', dpi=1200, bbox_inches='tight')
        plt.close()

        init = 0
        idx_rank = 1
        while init < total_abs_error*0.5:
            init += image4[ranked_indices_image_4[-idx_rank][0], ranked_indices_image_4[-idx_rank][1]]
            idx_rank += 1

        image_area = image4.shape[0] ** 2
        fraction_to_50 = idx_rank/image_area
        max_error = np.max(image4)
        print(f'Fraction of pixels that contribute to 50% of the error: {fraction_to_50}')
        print(f'Max error: {max_error}')

        np.savez(f'./fractions_removed/fraction_{fractions[0]}_channel_{channel}.npz', first=first, second=second, modified_img_total=modified_img_total, idx_rank=np.array(idx_rank), max_error=np.array(max_error))


        k = 0
        # plt.savefig(f'./images/one_percent_removal_pressure_yp_{config.WallRecon.TARGET_YP}_top_{fractions[0]}.png', dpi=1200, bbox_inches='tight')


        # scatter the replaced values to be better visible
        # axs[1].scatter(2*ranked_indices[-(int(0.01 * len(ranked_indices)) + 1):][:, 1],
        #             ranked_indices[-(int(0.01 * len(ranked_indices)) + 1):][:, 0], s=5, c='b', marker=',')



    # return index_u_v_w, first_np, second_total, mse_total, true_value, ranked_indices, sample_idx, pixel_frequency, fractions

# for fractions in [list(np.linspace(0, 1, 1000)), list(np.linspace(0, 0.1, 1000))]:
#     for index_uvw in range(3):
#         result_row = create_modified_prediction_continuous_pixels(sample_idx=0, pixel_frequency=100, index_u_v_w=index_uvw, fractions=fractions)
#         create_final_plot_single_pixel(result_row)

# for frac in tqdm(np.linspace(0, 1, 100)):
#     create_single_comparison_plot(sample_idx=0, pixel_frequency=100, index_u_v_w=0, fractions=[frac])










# first_np, second_np, model_img = create_modified_prediction()
# fig, axes = plt.subplots(nrows=3, ncols=2)
# for i in range(3):
#     ax = axes[i]
#     ax[0].imshow(first_np[i], cmap='coolwarm')
#     ax[1].imshow(second_np[i], cmap='coolwarm')
#
# axes[0][0].set_title('original prediction')
# axes[0][1].set_title(f'modified by removing top 1 pixel')