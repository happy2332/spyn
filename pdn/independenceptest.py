'''
Created on Oct 27, 2015

@author: molina
'''
#import igraph

from _random import Random
import os
import platform

if platform.system() == 'Darwin':
    os.environ["R_HOME"] = "/Library/Frameworks/R.framework/Resources/"
#else:
#    os.environ["R_HOME"] = "/usr/lib/R"
    
#print(os.environ["R_HOME"])

import numpy
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage



#from numba.typing.typeof import typeof
#import datasets
#from joblib import Memory
#import networkx as nx


path = os.path.dirname(__file__)
#memory = Memory(cachedir=path+"/cache", verbose=0)
    
with open (path+"/ptestglm.R", "r") as pdnfile:
    code = ''.join(pdnfile.readlines())
    ptestpdn = SignatureTranslatedAnonymousPackage(code, "ptestpdnglm")
 
 

# def getPtestpdnAdjacencyMatrix(data):
#     numpy2ri.activate()
#     try:
#         df = robjects.r["as.data.frame"](data)
#         out = ptestpdn.ptestglmblock(df)
#         #print(out)
#     except Exception as e:
#         #numpy.savetxt("/Users/alejomc/Dropbox/pspn/spyn/bin/data/graphlets/errordata.txt", data)
#         print(e)
#         print(data)
#         raise e 
#         
#     #print(out)
#     adjacency = numpy.array(out)
#     #print(adjacency)
#     
#     #print(adjacency.shape)
#     numpy2ri.deactivate()
#     return adjacency

#HIGHER ALPHA LOWER CHANCES OF HAVING A PROD NODE
# def getIndependentGroups(data, alpha):
# 
#     adjacency = getPtestpdnAdjacencyMatrix(data)
#     #print(adjacency)
#     #print(adjacency)
#     
#     adjacency[adjacency>alpha] = 0
#     adjacency[adjacency>0] = 1
#     
#     #print(adjacency)
#     
#     #0/0
#     
#     #TODO:make symmetric by hand
#     
#     
#     G = nx.Graph(adjacency)
#     
#     ccomponents = nx.connected_components(G)
#     #print(adjacency)
#     
#     cc = [c for c in sorted(ccomponents, key=len, reverse=True)]
#     
#     #print("sizes of components " + str(list(map(len,cc))))
#     
#     others = set(range(0,data.shape[1]))
#     others.difference_update(cc[0])
#     
#     
#     result = (list(cc[0]), list(others)) 
#     
#     print("partition " + str(result))
#     return result


def getPval(data):
    
    numpy2ri.activate()
    try:
        df = robjects.r["as.data.frame"](data)
        return ptestpdn.findpval(df)[0]
    except Exception as e:
        print(e)
        print(data)
        raise e 
        
    return 0.05
    


def getPtestpdnglm(data):
    
    numpy2ri.activate()
    try:
        #print("computing * node subset on instances, features: %s" % (data.shape,))
        df = robjects.r["as.data.frame"](data)
        out = ptestpdn.ptestpdnglm(df)
        out = numpy.asarray(out)
        #print(out)
    except Exception as e:
        #numpy.savetxt("/Users/alejomc/Dropbox/pspn/spyn/bin/data/graphlets/errordata.txt", data)
        print(e)
        print(data)
        raise e 

    return out

#from joblib import Memory
#memory = Memory(cachedir="/tmp", verbose=0)
#@memory.cache

def getIndependentGroups(data, alpha, families):
    
    numpy2ri.activate()
    try:
        #print("computing * node subset on instances, features: %s" % (data.shape,))
        #numpy.savetxt("/tmp/last.txt", data)
        df = robjects.r["as.data.frame"](data)
        out = ptestpdn.getIndependentGroupsAlpha(df, alpha, families)
        out = numpy.asarray(out)
        #print(out)
    except Exception as e:
        #numpy.savetxt("/Users/alejomc/Dropbox/pspn/spyn/bin/data/graphlets/errordata.txt", data)
        print(e)
        print(data)
        raise e 

    return out
 
    print(out)
    0/0
    out = numpy.asarray(out)
    result =  (numpy.where(out[:,0]>0)[0].tolist(),numpy.where(out[:,1]>0)[0].tolist())
    
    return result
    
    
    
if __name__ == '__main__':

    #data = numpy.loadtxt('/home/molina/Dropbox/Papers/pspn/spyn/bin/data/graphlets/out/wl/2ptc.build_wl_corpus.csv', dtype=int, delimiter=",")
    #data = numpy.loadtxt('/home/molina/Dropbox/Papers/pspn/spyn/bin/data/graphlets/out/wl/1mutag.build_wl_corpus.csv', dtype=int, delimiter=",")
    data = numpy.loadtxt('/tmp/last.txt', dtype=float)
    
    #print(data)
    out = getIndependentGroups(data, 0.05, "poisson")
    
    0/0
   
#    
#     data = numpy.array([
# [11, 2, 1],
# [11, 1, 2],
# [1, 1, 11],
# [2, 1, 11], ])
#     
#     #data = numpy.loadtxt(path+"/data.csv", dtype=int, delimiter=",", skiprows=1)
#     data = numpy.loadtxt('/home/molina/Dropbox/Papers/pspn/spyn/bin/data/graphlets/out/wl/1mutag.build_wl_corpus.csv', dtype=int, delimiter=",")
#     #data = numpy.loadtxt("/tmp/out.txt", dtype=float, delimiter=" ", skiprows=0)
#     adj = getPtestpdnAdjacencyMatrix(data)
#     
#     print(adj)
#     
#     indg = getIndependentGroups(data,0.1)
#     
#     print(indg)
#     
#     0/0
#     
#     print(data)
#     indg = getIndependentGroups(data)
#     0/0
#     adj = getPtestpdnAdjacencyMatrix(data)
#     print(adj)
