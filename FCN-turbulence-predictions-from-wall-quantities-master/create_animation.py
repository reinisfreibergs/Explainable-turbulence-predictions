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
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

yp = 15
data_shap = np.load(rf'final_shap_{yp}.npy')
data_x = np.load(rf'./x_test_{yp}.npy')
data_y = np.load(rf'./y_test_{yp}.npy')

channel = 0
first_channel_images = data_shap[0, :1000, channel, :, :]

# Create a figure and axis for the animation
fig, ax = plt.subplots()
ax.axis('off')
# Create an empty plot to be updated during animation
im = ax.imshow(first_channel_images[0], cmap='gray')

# Function to update the animation frames
def update(frame):
    im.set_array(first_channel_images[frame])
    return [im]

ani = FuncAnimation(fig, update, frames=range(len(first_channel_images)), interval=50)



Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

# Save the animation as an mp4 video
ani.save(f'u_first_channel_shap_animation.mp4', writer=writer)

plt.show()
k = 0


