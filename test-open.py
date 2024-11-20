import numpy as np
import matplotlib.pyplot as plt

new_data=np.load('test.npy')

# Display the result
plt.figure()
plt.imshow(new_data, extent=[0, new_data.shape[1], 0, new_data.shape[0]], origin='lower')
plt.title('Selected Region')
plt.show()