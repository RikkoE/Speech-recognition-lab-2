import proto2 as pro
import matplotlib.pyplot as plt
import numpy as np 
import tools2 as tool


example = np.load('lab2_example.npz')['example'].item()
tidigits = np.load('lab2_tidigits.npz')['tidigits']
models = np.load('lab2_models.npz')['models']

test = example['mfcc']

plt.imshow(test.T, interpolation = 'nearest', aspect = 'auto', origin = 'lower')
plt.colorbar()

plt.show()