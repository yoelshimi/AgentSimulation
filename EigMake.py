import numpy as np
from scipy.linalg import eig, eigh
from scipy.sparse.linalg import eigs, eigsh
import time

tic = time.perf_counter()

X = np.random.random((10000,10000))
print(time.perf_counter())

X = np.random.gamma(25.68946395563771, 10.33643812300134, (10000, 10000))
print(time.perf_counter())
#  X = np.random.random((10000,10000))
X = np.dot(X, X.T)

print(time.perf_counter())

evals_all, evecs_all = eigh(X)
print(time.perf_counter())

evals_all, evecs_all = eigsh(X, 3, which='LM')
print(time.perf_counter())

#  print(np.sort(evals_all))
#  print(time.perf_counter())
