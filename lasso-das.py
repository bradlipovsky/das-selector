from scipy.signal import butter, filtfilt
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

class SelectFromImage:
    def __init__(self, ax, image, vmin=None, vmax=None, cmap=None):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.image = image
        self.vmin = vmin
        self.vmax = vmax
        self.cmap = cmap
        self.mask = np.zeros(self.image.shape, dtype=bool)

        # Display the image with vmin, vmax, and cmap
        self.im = ax.imshow(self.image, extent=[0, self.image.shape[1], 0, self.image.shape[0]],
                            origin='lower', vmin=self.vmin, vmax=self.vmax, cmap=self.cmap,
                            aspect='auto')
        self.lasso = LassoSelector(ax, onselect=self.onselect)

    def onselect(self, verts):
        path = Path(verts)
        ny, nx = self.image.shape
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        points = np.vstack((x.flatten(), y.flatten())).T
        self.mask = path.contains_points(points).reshape(ny, nx)
        self.update()

    def update(self):
        # Apply the mask to the image
        masked_image = np.ma.array(self.image, mask=~self.mask)
        self.im.set_data(masked_image)
        self.ax.draw_artist(self.im)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()

'''
Load DAS data from an earthquake
'''

# Load DAS data
data_file = h5py.File('/auto/petasaur-wd15/rainier-10-14-2023-drive1/decimator_2023-08-27_10.10.00_UTC.h5', 'r')

# Select the data range without downsampling
#this_data = np.array(data_file['Acquisition/Raw[0]/RawData'][4850:5200, 1000:1100])
this_data = np.array(data_file['Acquisition/Raw[0]/RawData']) #crashes!
attrs = dict(data_file['Acquisition'].attrs)
data_file.close()

# Filter parameters
fs = 2 * attrs['MaximumFrequency']  # Sampling frequency (e.g., 200 Hz)
low_cut1 = 2
hi_cut1 = 10

# Apply bandpass filter
b, a = butter(2, (low_cut1, hi_cut1), btype='band', fs=fs)
data = filtfilt(b, a, this_data, axis=0)

# Decimate
data = data[4000:8000:10, :]

'''
Show the data and lasso it
'''

vm = 0.01

# Create the plot
fig, ax = plt.subplots()

# Add the SelectFromImage tool with vmin and vmax
selector = SelectFromImage(ax, data, vmin=-vm, vmax=vm, cmap='seismic')
plt.show()

# Extract the selected data
new_data = np.where(selector.mask, data, np.nan)

# Display the result
plt.figure()
plt.imshow(new_data, extent=[0, data.shape[1], 0, data.shape[0]], origin='lower',
           vmin=-vm, vmax=vm, cmap='seismic',aspect='auto')
plt.title('Selected Region')
plt.show()

np.save('test.npy', new_data)
