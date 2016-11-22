'''
Created on 24.02.2016

@author: alejomc
'''


from glob import glob
import hashlib
import locale
import locale
import math
import os
import sys
import time
sys.path.append("/Users/alejomc/Dropbox/pspn/spyn")
sys.path.append("/home/molina/Dropbox/papers/pspn/spyn")

import h2o
from joblib.memory import Memory
from matplotlib.backends.backend_pdf import PdfPages
from numba import cuda
from numba.types import float32
import numpy

from algo.learnspn import LearnSPN
import matplotlib.pyplot as plt
from mlutils import datasets
from mlutils.Plotter import plotStats, plotBoxes
from mlutils.benchmarks import Stats
from mlutils.datasets import removeOutliers
from mlutils.fastmath import compileEq
from mlutils.statistics import abs_error, squared_error
from mlutils.test import kfolded


locale.setlocale(locale.LC_NUMERIC, 'C')




#os.environ["NUMBAPRO_NVVM"] = "/usr/local/cuda-8.0/nvvm/lib64/libnvvm.so"
#os.environ["NUMBAPRO_LIBDEVICE"] = "/usr/local/cuda-8.0/nvvm/libdevice"



locale.setlocale(locale.LC_NUMERIC, 'C')


h2o.init()
h2o.no_progress()

def printlocal(string):
    with open("out.txt", "a") as myfile:
        myfile.write(str(string))
        myfile.write("\n")


memory = Memory(cachedir="/data/d1/molina/spn", verbose=0, compress=9)
# memory = Memory(cachedir="/tmp/spn", verbose=0)



@memory.cache
def spnComputeLambdas(spn, test):
    result = numpy.zeros(test.shape)
    
    for j in range(test.shape[1]):
        print(j,test.shape[1])
        instances = numpy.copy(test)
        instances[:, j] = None
        # print(instances.shape)
        result[:, j] = spn.complete(instances)[:, j]

    return result

@memory.cache
def spnComputeLambdas2(spn, test):
    result = numpy.zeros(test.shape)
    nF = test.shape[1]
    meq = spn.marginalizeToEquation(list(range(nF)))

    rep = {}
    for i in range(nF):
        rep["x_%s_" % i] = "x_%s_" % i
    
    f = compileEq(meq, rep)

    d = {}
    for j in range(nF):
        print(j,nF)
        x = numpy.arange(2 * numpy.max(test[:, j]))  # mean+variance
        
        for i in range(test.shape[0]):
            for p in range(nF):
                d["x_%s_" % p] = test[i, p]
            mv = 0
            for xval in x:
                d["x_%s_" % j] = xval
                feval = f(**d) 
                if feval > mv:
                    mv = feval
                    result[i, j] = xval
            

    return result


@cuda.jit('float32(float32, float32)', device=True, inline=False)
def cuda_poissonpmf(x, mean):
    return math.exp(-math.lgamma(x + 1.0) - mean + x * math.log(mean))



def executeSpnMaxJoint(eq, inp_array):
    nF = inp_array.shape[1]
    fid = hashlib.sha224(eq.encode('utf-8')).hexdigest()
    
    code = """
@cuda.jit("void(float32[:,:], float32[:,:], float32[:,:], int32[:])")
def spnmaxjoint_%s(data, outp, outv, maxv):
    i, j = cuda.grid(2)
    if i > data.shape[0] or j > data.shape[1]:
        return
    inp = cuda.local.array(%s, float32)
    for y in range(data.shape[1]):
        inp[y] = data[i, y]

    tmpprob = 0.0

    for val in range(maxv[j]):
        inp[j] = float(val)
        tmpprob = %s
        if outp[i, j] < tmpprob:
            outp[i, j] = tmpprob
            outv[i, j] = val
""" % (fid, nF, eq)
    print(code)
    print("compilation started")
    exec(code)
    print("compilation finished")

    inp_array = inp_array.astype(float)
    outp_array = numpy.zeros(inp_array.shape).astype(float)
    outv_array = numpy.zeros(inp_array.shape).astype(float)
    maxvals = (numpy.max(inp_array, axis=0)).astype(int)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(inp_array.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(inp_array.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    calls = "spnmaxjoint_%s[blockspergrid, threadsperblock](inp_array, outp_array, outv_array, maxvals)" % (fid)
    print("maxvals: ", maxvals)
    print("evaluating:",inp_array.shape," on ",calls)
    start = time.time()
    eval(calls)        
    print("in: ",(time.time() - start), " secs")
    return outv_array

@memory.cache
def spnComputeLambdasCuda(spn, data):
    eq = spn.marginalizeToEquation(list(range(data.shape[1])), fmt="cuda")
    
    return executeSpnMaxJoint(eq, data)

# @memory.cache
def pdnlearn(trainds, features, max_depth=100, iterations=20):
    #return 0
    from pdn.GBMPDN import GBMPDN
    pdn = GBMPDN(trainds, features, max_depth=max_depth, iterations=iterations)
    #pdn.addBoostIteration()
    #pdn.addBoostIteration()
    return pdn


# @memory.cache
def llspn(spn, test):
    return spn.eval(test)

# @memory.cache
def llpdn(pdn, test):
    return pdn.getLogLikelihood(test)



for dsname, data, featureNames in [datasets.getCommunitiesAndCrimes()]:

#for dsname, data, featureNames in [datasets.getNips(), datasets.getSynthetic(), datasets.getMSNBCclicks(), datasets.getCommunitiesAndCrimes()]:

    printlocal(dsname)
    printlocal(featureNames)
    printlocal(len(featureNames))
    printlocal(data.shape)
    
    
    stats = Stats(name=dsname)
    for train, test, i in kfolded(data, 5):
        spn = LearnSPN(alpha=0.001, min_instances_slice=80, cluster_prep_method="sqrt", cache=memory).fit_structure(train)

        printlocal("done")
        stats.addConfig("PSPN", spn.config)
        # stats.add("SPN Pois", Stats.LOG_LIKELIHOOD, llspn(spn, test))
        printlocal("LL")
        stats.add("PSPN", Stats.MODEL_SIZE, spn.size())
        printlocal("model size")
        prediction = spnComputeLambdas(spn, test)
        printlocal("model spnComputeLambdas")
        #prediction2 = spnComputeLambdasCuda(spn, test)
        prediction2 = spnComputeLambdas2(spn, test)
        printlocal("model spnComputeLambdas2")
        stats.add("PSPN", Stats.ABS_ERROR, abs_error(test, prediction))
        stats.add("PSPN", Stats.SQUARED_ERROR, squared_error(test, prediction))
        stats.add("PSPN_MJ", Stats.ABS_ERROR, squared_error(test, prediction2))
        stats.add("PSPN_MJ", Stats.SQUARED_ERROR, squared_error(test, prediction2))
        
        pdn = pdnlearn(train, featureNames, max_depth=30, iterations=20)
        stats.addConfig("PDN Pois", pdn.config)
        prediction = pdn.getLambdas(test)
        stats.add("PDN", Stats.MODEL_SIZE, pdn.size())
        print("MODEL SIZE PDN",pdn.size())
        stats.add("PDN", Stats.ABS_ERROR, abs_error(test, prediction))
        stats.add("PDN", Stats.SQUARED_ERROR, squared_error(test, prediction))
        printlocal("abs error")
        
        
        spn = spnlearn(numpy.log(train + 1), 0.001, 80, cluster_prep_method=None, family="gaussian")
        printlocal("done")
        stats.addConfig("GSPN", spn.config)
        stats.add("GSPN", Stats.MODEL_SIZE, spn.size())
        printlocal("model size")
        # stats.add("SPN Gaus", Stats.LOG_LIKELIHOOD, llspn(spn, numpy.log(test + 1)))
        printlocal("LL")
        prediction = numpy.exp(spnComputeLambdas(spn, test)) - 1
        stats.add("GSPN", Stats.ABS_ERROR, abs_error(test, prediction))
        stats.add("GSPN", Stats.SQUARED_ERROR, squared_error(test, prediction))
        printlocal("abs error")
    
                
        

    stats.save(dsname + "2.json")
    plotStats(stats, dsname + "2.pdf")
