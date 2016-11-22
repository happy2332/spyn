'''
Created on 15.07.2016

@author: alejomc
'''
import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models.hdpmodel import HdpModel
from joblib.memory import Memory
import math
import numpy
import warnings

from algo.learnspn import LearnSPN
from mlutils import datasets
from mlutils.benchmarks import Stats, Chrono
from mlutils.test import kfolded
from spn.linked.spn import Spn


memory = Memory(cachedir="/tmp/spn3", verbose=0, compress=9)



def runLda(corpus, dictionary, topics):
    return gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=topics, iterations=100, passes=10)


@memory.cache
def hldaperplexity(train, test):
    corpus = gensim.matutils.Dense2Corpus(train.astype(int), documents_columns=False)
    corpusTest = gensim.matutils.Dense2Corpus(test.astype(int), documents_columns=False)
    dictionary = Dictionary.from_corpus(corpus)
    
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Chrono().start()
        hlda = HdpModel(corpus, dictionary)
        c.end()

    corpus_words = sum(cnt for document in corpusTest for _, cnt in document)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ll = hlda.evaluate_test_corpus(corpusTest)
    perwordbound = ll / corpus_words
    print("LDA %.3f per-word bound, %.1f perplexity estimate based on a held-out corpus of %i documents with %i words" % 
                        (perwordbound, numpy.exp2(-perwordbound), len(corpusTest), corpus_words))
    return numpy.exp2(-perwordbound), c.elapsed()

@memory.cache
def ldaperplexity(train, test, topics):
    corpus = gensim.matutils.Dense2Corpus(train.astype(int), documents_columns=False)
    corpusTest = gensim.matutils.Dense2Corpus(test.astype(int), documents_columns=False)
    dictionary = Dictionary.from_corpus(corpus)
    
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = Chrono().start()
        lda = runLda(corpus, dictionary, topics=topics)
        c.end()

    corpus_words = sum(cnt for document in corpusTest for _, cnt in document)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perwordbound = lda.log_perplexity(corpusTest)
    print("LDA %.3f per-word bound, %.1f perplexity estimate based on a held-out corpus of %i documents with %i words" % 
                        (perwordbound, numpy.exp2(-perwordbound), len(corpusTest), corpus_words))
    return numpy.exp2(-perwordbound), c.elapsed()

@memory.cache
def pspnperplexity(train, test, min_slices, ind_test_method, row_cluster_method):
    c1 = Chrono().start()
    spn = LearnSPN(alpha=0.001, min_slices=min_slices, cluster_prep_method="sqrt", ind_test_method=ind_test_method, row_cluster_method=row_cluster_method).fit_structure(train)
    c1.end()
    time = c1.elapsed()
    pwb, perplexity, words, logl = spn.perplexity(test)
    
    print("SPN ll=%s %.3f per-word bound, %.1f perplexity estimate based on a held-out corpus of %i documents with %i words" % 
                    (logl, pwb, perplexity, test.shape[0], words))
    return perplexity, logl, time, spn.size()

@memory.cache
def pdnperplexity(train, test, features, depth, iters):
    from pdn.GBMPDN import GBMPDN
    c = Chrono().start()
    pdn = GBMPDN(train, features, max_depth=depth, iterations=iters)
    c.end()
    pwb, perplexity, words, ll = pdn.perplexity(test)
    print("PDN %s,%s=%s %.3f per-word bound, %.3f perplexity estimate based on a held-out corpus of %i documents with %i words" % 
                    (iters, depth, ll, pwb, perplexity, test.shape[0], words))
    return perplexity, ll, c.elapsed()

def filterDS(data, featureNames, no_below=5, no_above=0.3, keep_n=1200):
    corpus = gensim.matutils.Dense2Corpus(data, documents_columns=False)
    dictionary = Dictionary.from_corpus(corpus, {i: w for i, w in enumerate(featureNames)})
    
    if len(featureNames) > keep_n:
        dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    dw = dictionary.values()
    
    newWId = [i for i, w in enumerate(featureNames) if w in dw]
    words = [featureNames[i] for i in newWId]
    dictionary.compactify()
    data = data[:, newWId]
    return data, words

numpy.random.seed(1237)

for dsname, data, featureNames in [datasets.getNips(), datasets.getSynthetic(), datasets.getMSNBCclicks(), datasets.getCommunitiesAndCrimes()]:
   
    data, words = filterDS(data, featureNames)
    stats = Stats(name=dsname)
    nrfolds = 5
    for train, test, i in kfolded(data, nrfolds):

        print(dsname, train.shape, test.shape, i)
        
        for topics in [5, 10, 20, 50, 100]:
            stats.addConfig("LDA" + str(topics), {"topics":topics, "train documents":train.shape[0], "test documents":test.shape[0], "words":train.shape[1]})
            perplexity, tt = ldaperplexity(train, test, topics)
            stats.add("LDA" + str(topics), Stats.PERPLEXITY, perplexity)
            stats.add("LDA" + str(topics), Stats.TIME, tt)
            
        stats.addConfig("HLDA" + str(topics), {"topics":topics, "train documents":train.shape[0], "test documents":test.shape[0], "words":train.shape[1]})
        perplexity, tt = hldaperplexity(train, test)
        stats.add("HLDA" + str(topics), Stats.PERPLEXITY, perplexity)
        stats.add("HLDA" + str(topics), Stats.TIME, tt)
      
        for pct in [1, 10, 25, 50, 75, 90]:

            # use % in the leaves, the larger, the shallower
            min_slices = math.floor(((data.shape[0] / nrfolds) * (nrfolds - 1)) * pct / 100)
            
            
            stats.addConfig("PSPN" + str(pct), {"train documents":train.shape[0], "test documents":test.shape[0], "words":train.shape[1], "min_slices":min_slices})
            perplexity, ll, tt, size = pspnperplexity(train, test, min_slices, row_cluster_method="KMeans", ind_test_method="pairwise_treeglm")
            stats.add("PSPN" + str(100 - pct), Stats.PERPLEXITY, perplexity)
            stats.add("PSPN" + str(100 - pct), Stats.LOG_LIKELIHOOD, ll)
            stats.add("PSPN" + str(100 - pct), Stats.TIME, tt)
            stats.add("PSPN" + str(pct), Stats.MODEL_SIZE, size)
            
            
            stats.addConfig("PSPN_SAMPLED" + str(pct), {"train documents":train.shape[0], "test documents":test.shape[0], "words":train.shape[1], "min_slices":min_slices})
            perplexity, ll, tt, size = pspnperplexity(train, test, min_slices, row_cluster_method="RandomPartition", ind_test_method="subsample")
            stats.add("PSPN_SAMPLED" + str(100 - pct), Stats.PERPLEXITY, perplexity)
            stats.add("PSPN_SAMPLED" + str(100 - pct), Stats.LOG_LIKELIHOOD, ll)
            stats.add("PSPN_SAMPLED" + str(100 - pct), Stats.TIME, tt)
            stats.add("PSPN" + str(pct), Stats.MODEL_SIZE, size)
        
        print(train.shape)
        print(len(featureNames))
        if True:
            for iterations in [1, 5, 10, 25, 50, 100]:
                for depth in [8]:
                    stats.addConfig("PDN" + str(depth) + "_" + str(iterations), {"train documents":train.shape[0], "test documents":test.shape[0], "words":train.shape[1]})
                    perplexity, ll, tt = pdnperplexity(train, test, words, depth, iterations)
                    stats.add("PDN" + str(depth) + "_" + str(iterations), Stats.PERPLEXITY, perplexity)
                    stats.add("PDN" + str(depth) + "_" + str(iterations), Stats.LOG_LIKELIHOOD, ll)
                    stats.add("PDN" + str(depth) + "_" + str(iterations), Stats.TIME, tt)
            
        
        stats.save("stats9_" + dsname + ".json")
    stats.save("stats9_" + dsname + ".json")

