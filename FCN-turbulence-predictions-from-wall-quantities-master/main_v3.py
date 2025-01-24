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
# import cv2

# tf.config.run_functions_eagerly(True)

sys.path.insert(0, '../conf')
sys.path.insert(0, '../models')

# Training utils
from src.training_utils import load_trained_model
from src.tfrecord_utils import get_dataset
# from conf.config_sample import WallRecon
from sklearn.linear_model import LinearRegression
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# %% Configuration import
import config

prb_def = 'WallRecon'
# prb_def = os.environ.get('MODEL_CNN', None)
#
# if not prb_def:
#     print(
#         '"MODEL_CNN" enviroment variable not defined ("WallRecon" or "OuterRecon"), default value "WallRecon" is used')
#     app = config.WallRecon
#     prb_def = 'WallRecon'
# elif prb_def == 'WallRecon':
#     app = config.WallRecon
# elif prb_def == 'OuterRecon':
#     app = config.OuterRecon
# else:
#     raise ValueError('"MODEL_CNN" enviroment variable must be defined either as "WallRecon" or "OuterRecon"')

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

calculate_running_stats = False
if calculate_running_stats:
    total_sum_x = np.zeros((3,))
    total_squared_sum_x = np.zeros((3,))
    num_elements_x = np.zeros((3,))

    total_sum_y = np.zeros((3,))
    total_squared_sum_y = np.zeros((3,))
    num_elements_y = np.zeros((3,))

    for i in tqdm(range(n_samples_tot)):
        X_test = next(itrX).numpy()
        Y_test = np.array([np.array(i).squeeze() for i in next(itr)])

        total_sum_y += np.sum(Y_test, axis=(1, 2))
        total_squared_sum_y += np.sum(Y_test ** 2, axis=(1, 2))
        num_elements_y += np.prod(Y_test.shape[-1] ** 2)

        total_sum_x += np.sum(X_test, axis=(1, 2))
        total_squared_sum_x += np.sum(X_test ** 2, axis=(1, 2))
        num_elements_x += np.prod(X_test.shape[-1] ** 2)


    mean_x = total_sum_x / num_elements_x
    mean_x_sum = total_squared_sum_x / num_elements_x
    std_x = np.sqrt((mean_x_sum - mean_x**2))

    mean_y = total_sum_y / num_elements_y
    mean_y_sum = total_squared_sum_y / num_elements_y
    std_y = np.sqrt((mean_y_sum - mean_y**2))

    # print all means and stds
    print(mean_x, std_x, mean_y, std_y)
    np.save(rf'./mean_x_{app.TARGET_YP}.npy', mean_x)
    np.save(rf'./std_x_{app.TARGET_YP}.npy', std_x)
    np.save(rf'./mean_y_{app.TARGET_YP}.npy', mean_y)
    np.save(rf'./std_y_{app.TARGET_YP}.npy', std_y)
    Z = 0


X_test = np.ndarray((n_samples_tot, app.N_VARS_IN,
                     model_config['nx_'] + model_config['pad'],
                     model_config['nz_'] + model_config['pad']),
                    dtype='float')
Y_test = np.ndarray((n_samples_tot, app.N_VARS_OUT,
                     model_config['nx_'],
                     model_config['nz_']),
                    dtype='float')
ii = 0
for i in tqdm(range(n_samples_tot)):
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

X_test = np.load(rf'./x_test_{app.TARGET_YP}.npy')
Y_test = np.load(rf'./y_test_{app.TARGET_YP}.npy')
# Testing
Y_pred = np.ndarray((n_samples_tot, app.N_VARS_OUT,
                     model_config['nx_'], model_config['nz_']), dtype='float')

u_rms = model_config['rms'][0][model_config['ypos_Ret'][str(app.TARGET_YP)]]
default_scale = np.array([[(model_config['rms'][i][model_config['ypos_Ret'][str(app.TARGET_YP)]] / u_rms).numpy(),
                           avgs[i][model_config['ypos_Ret'][str(app.TARGET_YP)]].numpy()] for i in range(3)])

def default_scale_predicted_output(pred_init):
    # # Revert back to the flow field
    pred = np.array(pred_init)
    if app.SCALE_OUTPUT == True:
        u_rms = model_config['rms'][0] \
            [model_config['ypos_Ret'][str(app.TARGET_YP)]]

        for i in range(app.N_VARS_OUT):
            print('Rescale back component ' + str(i))
            pred[:, i] *= model_config['rms'][i][model_config['ypos_Ret'][str(app.TARGET_YP)]] / u_rms

    # do a comparison between the fluctuations so don't add the mean back

    # if app.FLUCTUATIONS_PRED == True:
    #     for i in range(app.N_VARS_OUT):
    #         print('Adding back mean of the component '+str(i))
    #         pred[:,i] = pred[:, i] + avgs[i][model_config['ypos_Ret'][str(app.TARGET_YP)]]

    return pred


CHECK_MODEL_CORRECTNESS = False
if CHECK_MODEL_CORRECTNESS:
    if app.N_VARS_OUT == 3:
        # reduce batch size for smaller gpu
        (Y_pred[:, 0, np.newaxis], Y_pred[:, 1, np.newaxis], Y_pred[:, 2, np.newaxis]) = \
            CNN_model.predict(X_test, batch_size=2)

    # np.min(Y_pred), np.max(Y_pred), np.min(Y_test), np.max(Y_test)
    total_comparisons = []
    Y_pred_copy = np.array(Y_pred)
    mse_total = []
    mse_orig_total = []
    for comparative_idx in range(3):

        final_mean, final_std = default_scale[comparative_idx][1], default_scale[comparative_idx][0]
        mse_orig = np.mean((np.array(Y_pred)[:, comparative_idx, :, :] - Y_test[:, comparative_idx, :, :])**2)
        # mse_default_scaled = np.mean((default_scale_predicted_output(np.array(Y_pred[:, comparative_idx, :, :])) - Y_test[:, comparative_idx, :, :])**2)
        mse_default_scaled = np.mean((final_std*(np.array(Y_pred[:, comparative_idx, :, :])) - Y_test[:, comparative_idx, :, :]) ** 2)

        reference_squared_value = np.mean(Y_test[:, comparative_idx, :, :]**2)
        comparisons = [reference_squared_value, mse_orig, mse_default_scaled]
        total_comparisons.append(comparisons)

        for i in tqdm(range(len(X_test))):

            final_mean, final_std = default_scale[comparative_idx][1], default_scale[comparative_idx][0]
            first_layer = tf.cast(final_std*CNN_model.output[comparative_idx][:, 0, :, :], dtype=tf.float64)
            # first_second_layer = first_layer+final_mean  # in case of predicting the whole field instead of just fluctuations
            output_mse_first = tf.math.square(tf.math.subtract(first_layer, Y_test[i][comparative_idx][None, :]))
            output_mse = tf.reduce_mean(output_mse_first, axis=(1, 2))

            output_mse = tf.cast(output_mse, dtype=tf.float64)  # important to recast to float64, otherwise amplified numerical errors
            first_output_model = Model(inputs=CNN_model.input, outputs=output_mse)

            # add extra first dim back to keep the shape
            Y_pred_next = first_output_model.predict(X_test[i][None, :])
            if i == 0:
                Y_pred_total = Y_pred_next
            else:
                Y_pred_total = np.concatenate([Y_pred_total, Y_pred_next])
            z = 0

        test_mse = mse_default_scaled - np.mean(Y_pred_total)
        mse_total.append(test_mse)

    assert np.mean(np.array(mse_total)) < 1e-10



# both are patches and should be examined why this version fails exactly
shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
# shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.linearity_1d(0)
final_shap = []
for comparative_idx in range(3):
    for i in tqdm(range(len(X_test))):

        final_mean, final_std = default_scale[comparative_idx][1], default_scale[comparative_idx][0]
        first_layer = tf.cast(final_std * CNN_model.output[comparative_idx][:, 0, :, :], dtype=tf.float64)
        # first_second_layer = first_layer + final_mean
        output_mse_first = tf.math.square(tf.math.subtract(first_layer, Y_test[i][comparative_idx][None, :]))
        output_mse = tf.reduce_mean(output_mse_first, axis=(1, 2))


        # first_output_model_v1 = Model(inputs=CNN_model.input, outputs=tf.reshape(CNN_model.output[0][:, 0, :, :], [-1, 192*192]))
        first_output_model = Model(inputs=CNN_model.input, outputs=output_mse)

        current_mean = np.broadcast_to(np.mean(np.mean(X_test[i], axis=-1, keepdims=True), axis=-2, keepdims=True),
                                       X_test[i].shape)

        # current_mean = np.broadcast_to(np.percentile(np.percentile(X_test[i], 75, axis=-1, keepdims=True), 75, axis=-2, keepdims=True), X_test[i].shape)

        # current_mean = cv2.GaussianBlur(X_test[i], (15, 15), 0)

        # pass as background the mean of this specific sample
        explainer = shap.DeepExplainer(first_output_model, current_mean[None, :])
        # explainer = shap.DeepExplainer(first_output_model, np.concatenate((X_test[:max(0, i-500)], X_test[min(i+901, 1000):])))

        # explain the specific X sample
        shap_values = explainer.shap_values(X_test[i][None, :], check_additivity=False)

        if i == 0 :
            shap_values_combined = shap_values
        else:
            shap_values_combined = np.concatenate((shap_values_combined, shap_values), axis=0)

    final_shap.append(shap_values_combined)

k = 0

# np.save(rf'final_shap_{app.TARGET_YP}_15.npy', np.array(final_shap))
k = 0
# np.save(rf'final_shap_{app.TARGET_YP}.npy', np.array(final_shap))
# np.save(rf'./x_test_{app.TARGET_YP}.npy', X_test)
# np.save(rf'./y_test_{app.TARGET_YP}.npy', Y_test)
# mean_shap = np.mean(np.abs(shap_values_combined), axis=0)
# plt.figure(figsize=(4, 8))
# for i, img in enumerate(mean_shap):
#     plt.subplot(len(mean_shap), 1, i+1)
#     plt.imshow(img, cmap='RdBu', vmin=np.min(img), vmax=np.max(img))
# plt.show()
#
# shap.image_plot([i.T for i in shap_values_combined], X_test.reshape(10, 208, 208, 3)[0])
#
#
#
# i_set_pred = 0
# while os.path.exists(pred_path + f'pred_fluct{i_set_pred:04d}.npz'):
#     i_set_pred = i_set_pred + 1
# print('[SAVING PREDICTIONS]')
# print('Saving predictions in ' + f'pred_fluct{i_set_pred:04d}')
# np.savez(pred_path + f'pred_fluct{i_set_pred:04d}', Y_test=Y_test, Y_pred=Y_pred)
