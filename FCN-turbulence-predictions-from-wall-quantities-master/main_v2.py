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

# %% Configuration import
import config

prb_def = 'WallReconfluct'
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



avg_inputs = np.broadcast_to(np.mean(np.mean(X_test, axis=-1, keepdims=True),
                                     axis=-2, keepdims=True),
                             X_test.shape)
CHECK_MODEL_CORRECTNESS = False
if CHECK_MODEL_CORRECTNESS:
    if app.N_VARS_OUT == 3:
        # reduce batch size for smaller gpu
        (Y_pred[:, 0, np.newaxis], Y_pred[:, 1, np.newaxis], Y_pred[:, 2, np.newaxis]) = \
            CNN_model.predict(X_test, batch_size=2)

    mse_total = []
    for comparative_idx in range(3):
        comparative_idx = 2
        mse_orig = np.mean((Y_pred[:, comparative_idx, :, :] - Y_test[:, comparative_idx, :, :])**2)

        for i in tqdm(range(len(X_test))):
            output_mse = tf.reduce_mean(tf.reduce_mean(tf.math.square(tf.math.subtract(CNN_model.output[comparative_idx][:, 0, :, :], Y_test[i][comparative_idx][None, :])), axis=-1), axis=-1)
            first_output_model = Model(inputs=CNN_model.input, outputs=output_mse)

            # add extra first dim back to keep the shape
            Y_pred_next = first_output_model.predict(X_test[i][None, :])
            if i == 0:
                Y_pred_total = Y_pred_next
            else:
                Y_pred_total = np.concatenate([Y_pred_total, Y_pred_next])
            z = 0

        test_mse = mse_orig - np.mean(Y_pred_total)
        mse_total.append(test_mse)

    assert np.mean(np.array(mse_total)) < 1e-8

#
# print(type(Y_pred))
# print(np.shape(Y_pred))
#
# # Revert back to the flow field
# if app.SCALE_OUTPUT == True:
#     u_rms = model_config['rms'][0] \
#         [model_config['ypos_Ret'][str(app.TARGET_YP)]]
#
#     for i in range(app.N_VARS_OUT):
#         print('Rescale back component ' + str(i))
#         Y_pred[:, i] *= model_config['rms'][i] \
#                             [model_config['ypos_Ret'][str(app.TARGET_YP)]] / \
#                         u_rms
#         Y_test[:, i] *= model_config['rms'][i] \
#                             [model_config['ypos_Ret'][str(app.TARGET_YP)]] / \
#                         u_rms

# if pred_fluct == True:
#     for i in range(app.N_VARS_OUT):
#         print('Adding back mean of the component '+str(i))
#         Y_pred[:,i] = Y_pred[:,i] + avgs[i][ypos_Ret[str(target_yp)]]
#         Y_test[:,i] = Y_test[:,i] + avgs[i][ypos_Ret[str(target_yp)]]

print(Y_pred.shape)

i_set_pred = 0
while os.path.exists(pred_path + f'pred_fluct{i_set_pred:04d}.npz'):
    i_set_pred = i_set_pred + 1
print('[SAVING PREDICTIONS]')
print('Saving predictions in ' + f'pred_fluct{i_set_pred:04d}')
np.savez(pred_path + f'pred_fluct{i_set_pred:04d}', Y_test=Y_test, Y_pred=Y_pred)