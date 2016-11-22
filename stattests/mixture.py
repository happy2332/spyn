'''
Created on Nov 12, 2015

@author: molina



'''
import gensim
import math
from numpy import float64
import numpy
import os
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr, SignatureTranslatedAnonymousPackage
from scipy.stats._continuous_distns import chi2
from scipy.stats._discrete_distns import poisson
from scipy.stats.contingency import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from statistics import mean

from pdn.datasets import getNips

from pdn.ABPDN import ABPDN

# with open ("/home/molina/Dropbox/pspn/spyn/pdn/pdn.R", "r") as pdnfile:
#    pdncode = '\n'.join(pdnfile.readlines())
#    pdnmodule = SignatureTranslatedAnonymousPackage(pdncode, "pdnmodule")
def optimizePDNs(data, ri):
    pdns = []
    for r in ri:
        d = r * data
        pdns.append(getPDN(d))
    return pdns

def getPDN(data):
    numpy2ri.activate()
    
    robjects.r["set.seed"](1337)
    
    df = robjects.r["as.data.frame"](data)
    try:
        out = pdnmodule.learnPDN(df, families="poisson", method="gbm")
    except Exception as e:
        numpy2ri.deactivate()
        print(e)
        print(data)
        raise e 
    
    return out

def computeMultivariatePoissonProbability(pdn, dataset):
    numpy2ri.activate()
    
    df = robjects.r["as.data.frame"](dataset)

    ev = pdnmodule.computeExpectedValues(pdn, df)
    print(ev)
    logprobs = pdnmodule.computeXlogProb(df, ev)
    print(logprobs)
    numpy2ri.deactivate()
    return logprobs




def getRi(pdns, W):
    nD = pdns[0].nD
    nM = len(W)
    ri = numpy.zeros((nD, nM))
    
    for i in range(0, nD):
        for m in range(0, nM):
            print((i, m, W[m], pdns[m].getLL(i), W[m] * pdns[m].getLL(i)))
            ri[i, m] = W[m] * pdns[m].getLL(i)
            
        ri[i, :] = ri[i, :] / sum(ri[i, :])
    print("RI")
    print(ri)
    return ri    

def trainPDN(data, ri):
    pdns = []
    for m in range(0, ri.shape[1]):
        ndata = numpy.multiply(data, ri[:, m][:, numpy.newaxis])
        ndata = numpy.round(ndata)
        print("ndata")
        print(ndata)
        pdn = ABPDN(ndata)
        pdn.addBoostIteration()
        # pdn.addBoostIteration()
        pdns.append(pdn)
        print("lambdas")
        print(pdn.lambdas)
    
    return pdns

def getKmeans(data, k=2):
    clustering = KMeans(n_clusters=2, n_init=10, max_iter=1000, random_state=1337, verbose=1)
    
    data2 = data
    
    s = numpy.sum(data2, axis=1)
    data2 = data2 / s[:, numpy.newaxis]
    # data2 = numpy.log(data2 + 1.0)
    print(data2)
    centers = clustering.fit_predict(data2, [0] * 50 + [1] * 50)
    print(sum(centers))
    print(centers)

def getInitialRI(data, nT, iterations=2):
    nD = data.shape[0]
    nW = data.shape[1]
    Z = numpy.zeros((nD, nT), dtype=float64)

    corpus = list(map(lambda doc: [(w, doc[w]) for w in range(0, nW) if doc[w] > 0], data.astype(int).tolist()))
        
    corpusDictionary = gensim.corpora.dictionary.Dictionary.from_corpus(corpus)
        
    lda = gensim.models.LdaModel(corpus=corpus, id2word=corpusDictionary, num_topics=nT, update_every=1, chunksize=nD, passes=iterations)
        
    for d in range(0, nD):
        for topic in lda[corpus[d]]:
            Z[d, topic[0]] = topic[1]
        
    topicWord = numpy.zeros((nT, nW))
    t = 0
    for topic in lda.show_topics(num_topics=nT, num_words=nW, formatted=False):
        for word in topic[1]:
            topicWord[t, int(word[1])] = float(word[0])
        
        t += 1
            
    return (Z, topicWord)    


def eval(z):
    print("section")
    ones = sum(z[0:50,])
    print("zeros %s\tones %s" %(50-ones,ones))
    ones = sum(z[50:100,])
    print("zeros %s\tones %s" %(50-ones,ones))



    

if __name__ == '__main__':
    
    data = numpy.loadtxt("/home/molina/Dropbox/Papers/pspn/spyn/pdn/mixtpdn.csv", dtype=float, delimiter=",", skiprows=1)
    
    
    
    #data = data / numpy.sum(data, axis=1)[:, numpy.newaxis]*10
    
    
    print(data)
    
    
    nM = 2
    numpy.random.seed(1337)
    z = ABPDN.pdnClustering(data,nM)
    print(z)
    eval(z)
    
    0 / 0
    # print(data)
    # getKmeans(data)
    
    # 0 / 0
    
    # ri = getInitialRI(data,2)[0]
    # #ri[ri>0.5] = 1
    # ri[ri<=0.5] = 0
    # print(ri)
    # print(sum(ri))

    # 0/0
    # data = getNips()[0]
    # data = numpy.array([[10, 1], [11, 2], [10, 2], [9, 1], [1, 10], [2, 11], [2, 10], [1, 9]])
    # data = numpy.genfromtxt('/home/molina/Dropbox/pspn/spyn/data/synthetic.ts.data', delimiter=',')
    # print(data)
    
    nD = data.shape[0]
    nF = data.shape[1]
    nM = 2

    wi = [0.5, 0.5]
    
    # ri = numpy.repeat([wi],nD,0)+numpy.random.rand(nD,nF)*0.1

    ri = getInitialRI(data, nM)[0]
    print(ri)

    
    pdns = trainPDN(data, ri)
    
    for i in range(0, 10):
        print(ri)
        ri = getRi(pdns, wi)
        print(ri)
        print("pre wi")
        print(wi)
        wi = numpy.sum(ri, axis=0) / nD
        print("pos  wi")
        print(wi)
        pdns = trainPDN(data, ri)
    
    print("results")
    print(data)
    print(ri)
    print(wi)
    print(pdns)
    print(numpy.round(ri, 1))
    
