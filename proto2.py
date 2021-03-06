import numpy as np
from tools2 import *

def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """

    rows, columns = log_emlik.shape
    
    res = 0

    for n in range(0, rows):
        res += logsumexp(np.log(weights) + log_emlik[n])

    return res

def forward(log_emlik, log_startprob, log_transmat):
    """Forward probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """

    rows, columns = log_emlik.shape

    alpha = np.zeros(shape=(rows, columns))

    alpha[0] = np.log(log_startprob) + log_emlik[0]

    log_transmat = log_transmat.T

    for n in range(1, rows):
        for j in range(0, columns):
            alpha[n][j] = logsumexp(alpha[n-1] + np.log(log_transmat[j])) + log_emlik[n][j]

    return alpha

def backward(log_emlik, log_startprob, log_transmat):
    """Backward probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """

def viterbi(log_emlik, log_startprob, log_transmat):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """

    rows, columns = log_emlik.shape

    vitVal = np.zeros(shape=(rows, columns))

    paths = np.zeros(shape=(rows, columns))

    vitVal[0] = np.log(log_startprob) + log_emlik[0]

    log_transmat = log_transmat.T

    for n in range(1, rows):
        for j in range(0, columns):
            vitVal[n][j] = np.amax(vitVal[n-1] + np.log(log_transmat[j])) + log_emlik[n][j]
            paths[n-1][j] = np.argmax(vitVal[n-1] + np.log(log_transmat[j]))
    
    #print "vitval: ", vitVal
    #print "paths: ", paths

    winner = np.argmax(vitVal[rows-1])
    winnerVal = np.amax(vitVal[rows-1])

    primePath = np.zeros((rows,), dtype = np.int)

    primePath[rows-1] = winner

    for i in range(rows-2, -1, -1):
        primePath[i] = paths[i][primePath[i+1]]

    res = [winnerVal, primePath]

    return res



