'''
Created on 14.06.2016

@author: alejomc
'''
from joblib.memory import Memory
import locale
from math import exp, lgamma, log
import mpmath
from numba import jit, vectorize, float32, float64, int32, int64
from numpy import PINF
import numpy
from scipy import NINF
from scipy.stats._continuous_distns import chi2
import sys
from scipy.stats import norm


locale.setlocale(locale.LC_NUMERIC, 'C')



#mpmath.mp.dps = 500


memory = Memory(cachedir="/tmp", verbose=False, compress=9)

@vectorize([float32(int32),
            float32(int64),
            float32(float32),
            float64(float64)])
def ufunclgamma(x):
    return lgamma(x + 1)
    
@jit(nopython=True, nogil=True)
def nplogpoissonpmf(x, mean):
    assert mean > 0
    #assert numpy.all(x >= 0)
    return -ufunclgamma(x) - mean + x * numpy.log(mean)

@jit(nopython=True, nogil=True)
def nppoissonpmf(x, mean):
    return numpy.exp(nplogpoissonpmf(x, mean))

@jit(nopython=True, nogil=True)
def poissonpmf(x, mean):
    return exp(logpoissonpmf(x, mean))

@jit(nopython=True, nogil=True)
def gaussianpdf(x, mean, variance):
    assert variance > 0, "gaussianpdf: variance needs to be > 0"
    result = numpy.exp(-((x - mean) ** 2) / (2.0 * variance)) / numpy.sqrt(2.0 * numpy.pi * variance)
    if result == 0:
        result = 0.000000000000001
    return result

@jit(nopython=True, nogil=True)
def loggaussianpdf(x, mean, variance):
    assert variance > 0, "loggaussianpdf: variance needs to be > 0"
    return numpy.log(gaussianpdf(x, mean, variance))

@jit(nopython=True, nogil=True)
def logpoissonpmf(x, mean):
    assert mean > 0, "logpoissonpmf: Mean needs to be > 0"
    assert x >= 0,  "logpoissonpmf: x needs to be >= 0"
    
    return -lgamma(x + 1) - float(mean) + float(x) * log(float(mean))

@jit(nopython=True, nogil=True)
def bernoullipmf(x, p):
    assert x == 0 or x ==1, "bernoullipmf: x needs to be 1/0"
    assert p > 0 or p <1, "bernoullipmf: p needs to be >0 or <1"
    return p**x * (1-p)**(1-x)

@jit(nopython=True, nogil=True)
def logbernoullipmf(x, p):
    return numpy.log(bernoullipmf(x, p))

def chi2cdf(x,k): 
    x,k = mpmath.mpf(x), mpmath.mpf(k) 
    #print(x,k)
    return mpmath.gammainc(k/2.0, 0.0, x/2.0, regularized=True)

def logchi2sf(x,k):
    
    res = chi2.logsf(x,k)
    for ix in numpy.where(res == NINF)[0]:
        res[ix] = float(mpmath.log(1.0-chi2cdf(x[ix],k[ix])))
    #print(res)
    #if res == NINF:
    #    res = float(mpmath.log(1.0-chi2cdf(x,k)))

    return res


def abs_error(test, predictions):
    return numpy.sum(numpy.abs(test - predictions)) / (test.shape[0] * test.shape[1])

def squared_error(test, predictions):
    return numpy.sum(numpy.power(test - predictions, 2)) / (test.shape[0] * test.shape[1])

if __name__ == '__main__':
    x = numpy.zeros((2, 2))
    x[0, :] = [1, 2]
    x[1, :] = [3, 4]
    
    print(logchi2sf(1654.234, 4.472670))
    0/0
    
    print(logpoissonpmf(1,1))
    
    0/0
    
    print(nppoissonpmf(x[:, 0], 2))
    
    print(poissonpmf(1, 2))
    print(poissonpmf(3, 2))
    
    
    print(lgamma(5 + 1))
    # should be close to: 4.787491742782045994247700934523243048399592315172032936009
    
    print(logpoissonpmf(2, 5))
    # should be close to -2.47427135569174456021571345500580128902429742582321981029
    
    # print(gaussianpdf(3.1, 4.5,1.5))
    # should be close to 0.1720518839354918


    print(gaussianpdf(0.0, 0.0229139563821 , 0.027269753061))
    
