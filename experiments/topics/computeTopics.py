'''
Created on 08.09.2016

@author: alejomc
'''
from joblib.memory import Memory
import numpy

from algo.learnspn import LearnSPN
from mlutils.datasets import getNips


memory = Memory(cachedir="/tmp/spn", verbose=0, compress=9)

@memory.cache
def spnlearn(data, alpha, min_slices=30, cluster_prep_method=None):
    
    numpy_rand_gen = numpy.random.RandomState(1337)
    
    print("learnspn")
    spn = LearnSPN(
        min_instances_slice=min_slices,
        row_cluster_method="KMeans",
        n_cluster_splits=2,
        # g_factor=5*10.0**-17,
        # g_factor=0.5,
        alpha=alpha,
        n_iters=2000,
        n_restarts=4,
        rand_gen=numpy_rand_gen,
        cluster_prep_method=cluster_prep_method).fit_structure(data=data)
    
    return spn

dsname, data, words = getNips()


spn = spnlearn(data, 0.1, 50)
spn.save_pdf_graph(words,outputfile="topics2a.pdf")