from sklearn.model_selection import train_test_split
import numpy as np
import os, fnmatch
import random
import sys

# dataDir = '/u/cs401/A3/data/'
dataDir = '../data .nosync'


class theta:
    def __init__(self, name, M=8, d=13):
        """Class holding model parameters.
        Use the `reset_parameter` functions below to
        initialize and update model parameters during training.
        """
        self.name = name
        self._M = M
        self._d = d
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))

    def precomputedForM(self, m):
        """Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma` (see slide 32)
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """
        M, d = self.mu.shape
        term1 = np.sum(np.square(self.mu[m]) / self.Sigma[m] / 2)
        term2 = d * (np.log(2 * np.pi)) / 2
        term3 = 0.5 * np.log(np.prod(self.Sigma[m]))
        return -(term1 + term2 + term3)

    def reset_omega(self, omega):
        """Pass in `omega` of shape [M, 1] or [M]
        """
        omega = np.asarray(omega)
        assert omega.size == self._M, "`omega` must contain M elements"
        self.omega = omega.reshape(self._M, 1)

    def reset_mu(self, mu):
        """Pass in `mu` of shape [M, d]
        """
        mu = np.asarray(mu)
        shape = mu.shape
        assert shape == (self._M, self._d), "`mu` must be of size (M,d)"
        self.mu = mu

    def reset_Sigma(self, Sigma):
        """Pass in `sigma` of shape [M, d]
        """
        Sigma = np.asarray(Sigma)
        shape = Sigma.shape
        assert shape == (self._M, self._d), "`Sigma` must be of size (M,d)"
        self.Sigma = Sigma


def log_b_m_x(m, x, myTheta):
    """ Returns the log probability of d-dimensional vector x using only
        component m of model myTheta (See equation 1 of the handout)

    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    Use `myTheta.preComputedForM(m)` for this.

    Return shape:
        (single row) if x.shape == [d], then return value is float (or equivalent)
        (vectorized) if x.shape == [T, d], then return shape is [T]

    You should write your code such that it works for both types of inputs.
    But we encourage you to use the vectorized version in your `train`
    function for faster/efficient computation.
    """
    sigma = myTheta.Sigma[m]
    mu = myTheta.mu[m]
    x_term = np.sum(-0.5 * (np.square(x) / sigma) + (np.multiply(mu, x) / sigma), axis=1)
    return x_term + myTheta.precomputedForM(m)


def log_p_m_x(log_Bs, myTheta):
    """ Returns the matrix of log probabilities i.e. log of p(m|X;theta)

    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)

    For further information, See equation 2 of handout

    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]

    NOTE: For a description of `log_Bs`, refer to the docstring of `logLik` below
    """
    log_omegabs = log_Bs + np.log(myTheta.omega)
    return log_omegabs - stable_logsumexp(log_omegabs, axis=0)


def logLik(log_Bs, myTheta):
    """ Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    """
    log_omegas = np.log(myTheta.omega)
    return np.sum(stable_logsumexp(log_Bs + log_omegas, axis=0))


def stable_logsumexp(array_like, axis=-1):
    """Compute the stable logsumexp of `vector_like` along `axis`
    This `axis` is used primarily for vectorized calculations.
    """
    array = np.asarray(array_like)
    # keepdims should be True to allow for broadcasting
    m = np.max(array, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(array - m), axis=axis))


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    """ Train a model for the given speaker. Returns the theta (omega, mu, sigma)"""
    myTheta = theta(speaker, M, X.shape[1])
    T, d = X.shape
    prev_L = float('-inf')
    improvement = float('inf')

    # perform initialization (Slide 32)
    myTheta.reset_omega(np.full((M, 1), 1 / M))
    myTheta.reset_mu(np.array([X[i] for i in np.random.choice(T, M)]))
    myTheta.reset_Sigma(np.full((M, d), 1.0))

    i = 0
    while i < maxIter and improvement >= epsilon:

        # compute intermediate results
        log_Bs = np.zeros((M, T))
        for m in range(M):
            log_Bs[m] = log_b_m_x(m, X, myTheta)
        log_Ps = log_p_m_x(log_Bs, myTheta)
        L = logLik(log_Bs, myTheta)

        # Update Parameters
        log_Ps_exp = np.exp(log_Ps)

        p_m_x = np.sum(log_Ps_exp, axis=1).reshape((M, 1))
        myTheta.omega = p_m_x / float(T)

        myTheta.mu = np.divide(np.dot(log_Ps_exp, X), p_m_x)

        myTheta.Sigma = np.divide(np.dot(log_Ps_exp, X ** 2), p_m_x) - (myTheta.mu ** 2)

        improvement = L - prev_L
        prev_L = L
        i = i + 1

    return myTheta


def test(mfcc, correctID, models, k=5):
    """ Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    """
    bestModel = -1
    log_likelihood = np.zeros(len(models))
    T, d = mfcc.shape
    M = models[0].omega.shape[0]
    output = []
    log_Bs = np.zeros((len(models), M, T))
    for i in range(len(models)):
        for m in range(M):
            log_Bs[i, m, :] = log_b_m_x(m, mfcc, models[i])

    for i in range(len(models)):
        log_likelihood[i] = logLik(log_Bs[i], models[i])

    bestModel = np.argmax(log_likelihood)
    if k > 0:
        top_k = log_likelihood.argsort()
        print('{}'.format(models[correctID].name))
        output.append('{}'.format(models[correctID].name))
        for i in range(1, k + 1):
            print('{} {}'.format(models[int(top_k[-i])].name, log_likelihood[int(top_k[-i])]))
            output.append('{} {}'.format(models[int(top_k[-i])].name, log_likelihood[int(top_k[-i])]))

    # fout = open("gmmLiks.txt", 'a')
    # for line in output:
    #     fout.write(line)
    #     fout.write("\n")
    # fout.close()
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    # print("TODO: you will need to modify this main block for Sec 2.3")
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*npy")
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))

            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate
    numCorrect = 0

    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
    print("Accuracy: {}".format(accuracy))
    sys.stdout = open("gmmLiks.txt", "w")
    # fout = open("gmmLiks.txt", 'a')
    # fout.write("\nAccuracy: {}".format(accuracy))
    # fout.close()
