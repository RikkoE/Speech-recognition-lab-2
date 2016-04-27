import proto2 as pro
import matplotlib.pyplot as plt
import numpy as np 
from tools2 import *

def gmmscore(model, tidigits):

    res = np.zeros(shape=(44, 11))

    for j in range(0, 44):
        tid = tidigits[j]['mfcc']
        for i in range(0, 11):
            modelGmm = model[i]['gmm']

            cvGmm = modelGmm['covars']
            muGmm = modelGmm['means']

            gmmobs = log_multivariate_normal_density_diag(tid, muGmm, cvGmm)

            weights = modelGmm['weights']

            res[j][i] = pro.gmmloglik(gmmobs, weights)

    return res

def gmmMinimum(scorematrix):
    gmmwinner = scorematrix

    for i in range(0, 44):
        minimum = np.argmax(scorematrix[i])
        gmmwinner[i] = 0
        gmmwinner[i][minimum] = -4000

    return gmmwinner

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

exhmmalpha = example['hmm_logalpha']

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

#print "Exgmmlog: ", exgmmlog
#print "Gmm log lik: ", gloglik


#scorematrix = gmmscore(models, tidigits)
#gmmwinner = gmmMinimum(scorematrix)


################################################
#           HMM log likelihood                 #
################################################

startprob = modelHmm['startprob']
transmat = modelHmm['transmat']

#logalpha = pro.forward(hmmobs, startprob, transmat)

#ALPHA FORWARD HMM
alpha = pro.forward(hmmobs, startprob, transmat)


################################################
#               Plot results                   #
################################################

#plt.plot(scorematrix)

#plt.imshow(exhmmalpha.T, interpolation = 'nearest', aspect = 'auto', origin = 'lower')
#plt.colorbar()


ax = plt.subplot(2, 1, 1)
ax.imshow(exhmmalpha.T, interpolation = 'nearest', aspect = 'auto', origin = 'lower')

ax = plt.subplot(2, 1, 2)
ax.imshow(alpha.T, interpolation = 'nearest', aspect = 'auto', origin = 'lower')

plt.show()