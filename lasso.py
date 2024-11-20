import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

class SelectFromImage:
    def __init__(self, ax, image):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.image = image
        self.mask = np.zeros(self.image.shape, dtype=bool)

        # Display the image
        self.im = ax.imshow(self.image, extent=[0, self.image.shape[1], 0, self.image.shape[0]], origin='lower')
        self.lasso = LassoSelector(ax, onselect=self.onselect)

    def onselect(self, verts):
        path = Path(verts)
        ny, nx = self.image.shape
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        points = np.vstack((x.flatten(), y.flatten())).T
        self.mask = path.contains_points(points).reshape(ny, nx)
        self.update()

    def update(self):
        masked_image = np.ma.array(self.image, mask=~self.mask)
        self.ax.clear()
        self.ax.imshow(masked_image, extent=[0, self.image.shape[1], 0, self.image.shape[0]], origin='lower')
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()

# Sample data
data = np.random.rand(200, 300)

fig, ax = plt.subplots()
selector = SelectFromImage(ax, data)
plt.show()

# Extract the selected data
new_data = np.where(selector.mask, data, np.nan)

# Display the result
plt.figure()
plt.imshow(new_data, extent=[0, data.shape[1], 0, data.shape[0]], origin='lower')
plt.title('Selected Region')
plt.show()

np.save('test.npy', new_data)

exit()