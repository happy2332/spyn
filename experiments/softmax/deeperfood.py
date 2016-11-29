'''
Created on Nov 17, 2016

@author: molina
'''
import time

from cvxpy import *
from joblib.memory import Memory
import numpy
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics.ranking import roc_curve, auc

from algo.learnspn import LearnSPN
from experiments.graphclassification.spngraphlet import SPNClassifier
import matplotlib.pyplot as plt


numpy.set_printoptions(threshold=numpy.inf)

memory = Memory(cachedir="/data/d1/happy/spyn/experiments/softmax", verbose=0, compress=9)

def getTopIndices(arr):
    # sort logits descanding
    # There is no direct method of getting argsort descending, so first 
    # sort in increasing, then reverse the list 
    np_sorted_logits_asc = numpy.argsort(arr,1)
    np_sorted_logits_desc = []
    for _, e in enumerate(np_sorted_logits_asc):
        np_sorted_logits_desc.append(e[::-1])
    return np_sorted_logits_desc

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    # print(predicted)
    # print(actual)

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    # print(score)
    # if not actual:
    #    return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return numpy.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def getOneHotEncoding(data, vocab_size=100):
    result = numpy.zeros((data.shape[0], vocab_size))
    
    result[list(range(data.shape[0])), data] = 1
    return result


def spnClassificationNBPred(model, X):
    
    trainll = numpy.zeros((X.shape[0], model['classes'].shape[0]))
    
    for j, spn in enumerate(model["spns"]):
        trainll[:, j] = spn.eval(X, individual=True) + numpy.log(model['weights'][j])

    pmax = numpy.argmax(trainll, axis=1)

    return (model['classes'][pmax], trainll)



def spnClassificationNBFit(X, Y, alpha=0.001, min_slices=80):
    classes = numpy.unique(Y)
    spns = []
    
    # trainll = numpy.zeros((X.shape[0],classes.shape[0]))
    ws = []
    for j in range(classes.shape[0]):
        idx = Y == classes[j]
        ws.append(float(numpy.sum(idx)) / X.shape[0])
        
        data_train_class = X[idx, :]
        spn = LearnSPN(cache=memory, alpha=alpha, min_instances_slice=min_slices, cluster_prep_method=None, families="gaussian").fit_structure(data_train_class)
        spns.append(spn)
        
        # trainll[idx, j] = spn.eval(data_train_class, individual=True)
        

    return {'classes':classes, 'spns':spns, 'weights':ws}

def spnClassificationGeneralFit(X, Y, maxClasses, alpha=0.001, min_slices=500):
    # need to convert Y into one-hot encoding as there is no multinomial till now
    #Y = getOneHotEncoding(Y, maxClasses)
    print('X shape : ',X.shape)
    print('Y shape : ',Y.shape)
    families = ['gaussian']*X.shape[1]+['binomial']*Y.shape[1]
    data_train_class = numpy.c_[X,Y]
    spn = LearnSPN(cache=memory, row_cluster_method="RandomPartition",ind_test_method="subsample",alpha=alpha, min_features_slice=30, min_instances_slice=min_slices, cluster_prep_method=None, families=families).fit_structure(data_train_class)
    return spn

def spnClassificationGeneralPred(model, X, maxClasses):
    print("Total number of instances : ",X.shape[0])
    predictions = numpy.zeros((X.shape[0],maxClasses))
    for i,feature in enumerate(X):
        if(i%100 == 0):
            print('Number of instances completed : ',i)
        loglikelihood = numpy.zeros(maxClasses)
        for j in range(maxClasses):
            Y = numpy.zeros(maxClasses)
            Y[j] = 1
            data = numpy.transpose(numpy.r_[feature,Y])
            ll = model.eval(data, individual=True)
            loglikelihood[j] = ll[0]
        predictions[i] = loglikelihood
    return (predictions)


if __name__ == '__main__':
    in_train = numpy.loadtxt("data/food/train/_preLogits.csv", delimiter=",")
    out_train = numpy.loadtxt("data/food/train/_groundtruth.csv", delimiter=",").astype(int)
    
    in_test = numpy.loadtxt("data/food/validation/_preLogits.csv", delimiter=",")
    out_test = numpy.loadtxt("data/food/validation/_groundtruth.csv", delimiter=",").astype(int)
    
    
    # Get indices of positive labels in ground truth
    groundTruth_labels = [numpy.nonzero(i)[0] for i in out_test]
        
    nnpred_test = numpy.loadtxt("data/food/validation/_predictions.csv", delimiter=",")
    
    # sort logits descanding
    # There is no direct method of getting argsort descending, so first 
    # sort in increasing, then reverse the list 
    np_sorted_logits_asc = numpy.argsort(numpy.loadtxt("data/food/validation/_logits.csv", delimiter=","), 1)
    np_sorted_logits_desc = []
    for _, e in enumerate(np_sorted_logits_asc):
        np_sorted_logits_desc.append(e[::-1])

    mAP_4 = mapk(groundTruth_labels, np_sorted_logits_desc, k=4)
    print('mAP_4: %f' % mAP_4)
    mAP_6 = mapk(groundTruth_labels, np_sorted_logits_desc, k=6)
    print('mAP_6: %f' % mAP_6)
    mAP_8 = mapk(groundTruth_labels, np_sorted_logits_desc, k=8)
    print('mAP_8: %f' % mAP_8)
    mAP_10 = mapk(groundTruth_labels, np_sorted_logits_desc, k=10)
    print('mAP_10: %f' % mAP_10)
    mAP_15 = mapk(groundTruth_labels, np_sorted_logits_desc, k=15)
    print('mAP_15: %f' % mAP_15)
    mAP_20 = mapk(groundTruth_labels, np_sorted_logits_desc, k=20)
    print('mAP_20: %f' % mAP_20)
    mAP_30 = mapk(groundTruth_labels, np_sorted_logits_desc, k=30)
    print('mAP_30: %f' % mAP_30)
    mAP_41 = mapk(groundTruth_labels, np_sorted_logits_desc, k=41)
    print('mAP_41: %f' % mAP_41)

    numClasses = out_train.shape[1]
    start_time = time.time()
    spnmodel = spnClassificationGeneralFit(in_train,out_train,numClasses,min_slices=100)
    print("Learning SPN took %s seconds ---\n" % (time.time() - start_time))
    print("Testing Deep SPN...\n")
    start_time = time.time()
    spnPred = spnClassificationGeneralPred(spnmodel,in_test,numClasses)
    print("Testing SPN took %s seconds ---\n" % (time.time() - start_time))
    spnPredLabels = getTopIndices(spnPred)
    spnmAP_4 = mapk(groundTruth_labels, spnPredLabels, k=4)
    print('mAP_4: %f' % spnmAP_4)
    
