import numpy as np
import matplotlib.pyplot as plt

new_data=np.load('test.npy')

# Display the result
plt.subplots(2,1)
plt.subplot(211)
plt.imshow(new_data, extent=[0, new_data.shape[1], 0, new_data.shape[0]], origin='lower',
            aspect='auto')
plt.title('Selected Region')

plt.subplot(212)
channel_max = np.nanmax( np.abs(new_data),axis=0)
plt.plot(np.log10(channel_max))

plt.savefig("test.png")
