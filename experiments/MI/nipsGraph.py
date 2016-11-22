'''
Created on Jul 27, 2016

@author: molina
'''
from joblib.memory import Memory
import numpy

from algo.learnspn import LearnSPN
from mlutils.datasets import getNips
import networkx as nx
import matplotlib.pyplot as plt


memory = Memory(cachedir="/data/d1/molina/spn", verbose=0, compress=9)



@memory.cache
def getMIAdjc(spn, words):
    nF = len(words)
    adj = numpy.zeros((nF,nF))
    
    for i in range(nF):
        for j in range(i+1, nF):
            print(i,j)
            adj[i,j] = adj[j,i] = spn.computeMI(words[i], words[j], words, True)
    
    return adj

@memory.cache
def getDistAdjc(spn, words):
    nF = len(words)
    adj = numpy.zeros((nF,nF))
    
    for i in range(nF):
        for j in range(i+1, nF):
            print(i,j)
            adj[i,j] = adj[j,i] = spn.computeDistance(words[i], words[j], words, True)
    
    return adj


dsname, data, words = getNips()


spn = LearnSPN(alpha=0.001, min_instances_slice=100, cluster_prep_method="sqrt", cache=memory).fit_structure(data)


adjc = getMIAdjc(spn, words)

#adjc = getDistAdjc(spn, words)

adjc = numpy.log(adjc)

print(adjc)
print(numpy.any(adjc > 0.8))


def show_graph_with_labels(fname, adjacency_matrix, mylabels):
    def make_label_dict(labels):
        l = {}
        for i, label in enumerate(labels):
            l[i] = label
        return l
    
    gr = nx.Graph(adjacency_matrix)
    
    for k,v in make_label_dict(mylabels).items():
        gr.node[k]["label"] = v

    nx.draw(gr, node_color='#36C8FF',edge_color='#36C8FF', node_size=1, labels=make_label_dict(mylabels), with_labels=True)
    nx.write_gexf(gr, "graphlogMI.gexf")
    nx.relabel_gexf_graph(gr)
    #plt.show()
    #plt.savefig(fname)


show_graph_with_labels("graphlogMI.pdf", adjc, words)
