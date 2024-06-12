import numpy as np
import os
import matplotlib.pyplot as plt
import platform
if platform.system() == 'Windows':
    import matplotlib
    matplotlib.use("qt5agg")
from matplotlib.ticker import ScalarFormatter
from skimage.transform import resize
from scipy.ndimage import label
from tqdm import tqdm
import pandas as pd

'''
..
'''


yp = 15
data_shap = np.load(rf'final_shap_{yp}.npy')
data_x = np.load(rf'./x_test_{yp}.npy')
data_y = np.load(rf'./y_test_{yp}.npy')


