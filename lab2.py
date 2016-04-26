import proto2 as pro
import matplotlib.pyplot as plt
import numpy as np 
from tools2 import *

################################################
#        Load necessary variables              #
################################################

example = np.load('lab2_example.npz')['example'].item()
tidigits = np.load('lab2_tidigits.npz')['tidigits']
models = np.load('lab2_models.npz')['models']

exmfcc = example['mfcc']

exhmmobs = example['hmm_obsloglik']
exgmmobs = example['gmm_obsloglik']

exgmmlog = example['gmm_loglik']

################################################
#         HMM Observed log likelihood          #
################################################

modelHmm = models[0]['hmm']
modkeyHmm = modelHmm.keys()

cvHmm = modelHmm['covars']
muHmm = modelHmm['means']

hmmobs = log_multivariate_normal_density_diag(exmfcc, muHmm, cvHmm)

################################################
#        GMM Observed log likelihood           #
################################################

modelGmm = models[0]['gmm']
modkeyGmm = modelGmm.keys()

cvGmm = modelGmm['covars']
muGmm = modelGmm['means']

gmmobs = log_multivariate_normal_density_diag(exmfcc, muGmm, cvGmm)

################################################
#           GMM log likelihood                 #
################################################

weights = modelGmm['weights']

gloglik = pro.gmmloglik(gmmobs, weights)

print "Exgmmlog: ", exgmmlog
print "Gmm log lik: ", gloglik

################################################
#               Plot results                   #
################################################


#ax = plt.subplot(2, 1, 1)
#ax.imshow(exgmmobs.T, interpolation = 'nearest', aspect = 'auto', origin = 'lower')

#ax = plt.subplot(2, 1, 2)
#ax.imshow(gmmobs.T, interpolation = 'nearest', aspect = 'auto', origin = 'lower')

plt.show()