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

data_shap = np.load(r'final_shap_100_v3.npy')
data_x = np.load(r'./x_test_100.npy')
data_y = np.load(r'./y_test_100.npy')


i = 0

thresh = 0.99
k = 0
channel = 0

linear_fit_coefs_yp15 = np.array([[1.0067024127808921, 0.28081194650770736],
                    [1.995712269446302, 0.00113387769428069],
                    [0.8699978550994905, 0.0007984776097612707]])


def create_modified_prediction(channel=2, i=0):

    # v1 - remove the n-th most important pixel and track the change in mse.
    ranked_indices = np.array(np.unravel_index(np.argsort(np.abs(data_shap)[0][i][channel].flatten()), data_shap[0][i][channel].shape)).T
    current_mean = np.mean(X_test[i], axis=(-1, -2))

    true_value = np.array(Y_test[i])


    # use every 100th for now as every single one will be too much
    for pix_idx in range(0, len(ranked_indices), 500):
        # it sorts starting from smallest so select from back
        modified_img = np.array(X_test[i])
        modified_img[channel][ranked_indices[-(pix_idx+1)][0], ranked_indices[-(pix_idx+1)][1]] = current_mean[channel]
        modified_img = modified_img[None, :]
        if pix_idx == 0:
            modified_img_total = modified_img
        else:
            modified_img_total = np.concatenate((modified_img_total, modified_img))
        # modified_img[channel][ranked_indices[pix_idx][0], ranked_indices[pix_idx][1]] = current_mean[0]

    # modified_img[channel][(data_shap[0][i][channel] < (1-thresh)*np.max(data_shap[0][i][channel])) == 0] = current_mean[0]

    # plt.imshow(data_x[i][0], cmap='coolwarm')
    # plt.figure()
    # plt.imshow(modified_img, cmap='coolwarm')

    # return the batch dim for model
    # modified_img = modified_img[None, :]


    first = CNN_model.predict(data_x[i][None, :])
    second = CNN_model.predict(modified_img_total, batch_size=2)

    a, b = linear_fit_coefs_yp15[:, 0][:, None, None, None], linear_fit_coefs_yp15[:, 1][:, None, None, None]
    first_np = a * np.concatenate(first, axis=1) + b
    second_np = a * np.concatenate(second, axis=1) + b

    first_loss = np.mean((true_value - first_np)**2)
    second_loss = np.mean((true_value - second_np)**2)


    return first_np, second_np, modified_img

first_np, second_np, model_img = create_modified_prediction()

fig, axes = plt.subplots(nrows=3, ncols=2)
for i in range(3):
    ax = axes[i]
    ax[0].imshow(first_np[i], cmap='coolwarm')
    ax[1].imshow(second_np[i], cmap='coolwarm')

axes[0][0].set_title('original prediction')
axes[0][1].set_title(f'modified by removing top 1 pixel')