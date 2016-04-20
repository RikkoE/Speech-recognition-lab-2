import proto2 as pro
import matplotlib.pyplot as plt
import numpy as np 
import tools2 as tool
from sklearn import mixture


example = np.load('lab2_example.npz')['example'].item()
tidigits = np.load('lab2_tidigits.npz')['tidigits']
models = np.load('lab2_models.npz')['models']

exmfcc = example['mfcc']
exhmmobs = example['hmm_obsloglik']
exgmmobs = example['gmm_obsloglik']

modelHmm = models[0]['hmm']
modkeyHmm = modelHmm.keys()

cvHmm = modelHmm['covars']
muHmm = modelHmm['means']

hmmobs = mixture.log_multivariate_normal_density(exmfcc, muHmm, cvHmm, 'diag')


modelGmm = models[0]['gmm']
modkeyGmm = modelGmm.keys()

cvGmm = modelGmm['covars']
muGmm = modelGmm['means']

gmmobs = mixture.log_multivariate_normal_density(exmfcc, muGmm, cvGmm, 'diag')


ax = plt.subplot(2, 1, 1)
ax.imshow(exgmmobs.T, interpolation = 'nearest', aspect = 'auto', origin = 'lower')

ax = plt.subplot(2, 1, 2)
ax.imshow(gmmobs.T, interpolation = 'nearest', aspect = 'auto', origin = 'lower')

plt.show()