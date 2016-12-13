'''
Created on Dec 7, 2016

@author: molina
'''
from joblib.memory import Memory
import numpy

from algo.learnspn import LearnSPN


if __name__ == '__main__':
    
    
    memory = Memory(cachedir="/tmp", verbose=0, compress=9)

    #data = numpy.loadtxt("data/breast_cancer/wdbc.data", delimiter=",")
    #data = data[:,1:]
    
    features_data = numpy.loadtxt("data/food/train/_preLogits.csv", delimiter=",")
    labels_data = numpy.loadtxt("data/food/train/_groundtruth.csv", delimiter=",").astype(int)
    data = numpy.c_[features_data,labels_data]
    print(data.shape)
    print(data[1,:])
    
    fams = ["gaussian"]*features_data.shape[1] + ["binomial"]*labels_data.shape[1]
    spn = LearnSPN(cache=memory, alpha=0.001, min_instances_slice=200, cluster_prep_method=None, families=fams).fit_structure(data)
    
    
    
    print(spn.to_tensorflow(["V"+str(i) for i in range(data.shape[1])], data))
    