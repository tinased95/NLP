import a3_gmm_structured
import numpy as np

d = 13
k = 5  # number of top speakers to display, <= 0 if none
M = 8
T = 100

m = 1
X = np.random.rand(T, d)
log_Bs = np.ones((M, T))

myTheta = a3_gmm_structured.theta('S-13A', M, d)
myTheta.Sigma = np.ones((M, d))
myTheta.precomputedForM(m)
x_term = a3_gmm_structured.log_b_m_x(m, X, myTheta)
print(x_term.shape)
# loglik = a3_gmm_structured.logLik(log_Bs, myTheta)
# myTheta = a3_gmm_structured.train('speaker', X)
# print(myTheta.mu)