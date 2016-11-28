'''
Created on Nov 17, 2016

@author: molina
'''
import time

from cvxpy import *
from joblib.memory import Memory
import numpy
from sklearn import ensemble, naive_bayes
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics.classification import accuracy_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB

from algo.learnspn import LearnSPN
import matplotlib.pyplot as plt
from mlutils.test import classkfolded


numpy.set_printoptions(threshold=numpy.inf)

memory = Memory(cachedir="/tmp", verbose=0, compress=9)

def getOneHotEncoding(data, vocab_size=100):
    result = numpy.zeros((data.shape[0], vocab_size))
    
    result[list(range(data.shape[0])),data] = 1
    return result

def convert_to_contigous(numbers):
    _,indices = numpy.unique(numbers,return_inverse=True)
    return indices

def spnClassificationNBPred(model, X):
    
    trainll = numpy.zeros((X.shape[0],model['classes'].shape[0]))
    
    for j, spn in enumerate(model["spns"]):
        trainll[:, j] = spn.eval(X, individual=True) + numpy.log(model['weights'][j])

    pmax = numpy.argmax(trainll, axis=1)
    #print('pmax : ',pmax)
    return (model['classes'][pmax], trainll)



def spnClassificationNBFit(X, Y, alpha=0.001, min_slices=500):
    min_slices = X.shape[0]+1
    classes = numpy.unique(Y)
    #print(Y.shape)
    #print(classes)
    spns = []
    
    trainll = numpy.zeros((X.shape[0],classes.shape[0]))
    ws = []
    for j in range(classes.shape[0]):
        idx = Y == classes[j]
        ws.append(float(numpy.sum(idx))/X.shape[0])
        
        data_train_class = X[idx, :]
        spn = LearnSPN(cache=memory, alpha=alpha, min_features_slice=10, row_cluster_method="RandomPartition",ind_test_method="subsample", min_instances_slice=min_slices, cluster_prep_method=None, families="gaussian").fit_structure(data_train_class)
        spns.append(spn)
        
        trainll[idx, j] = spn.eval(data_train_class, individual=True)
        

    x = Variable(len(classes))
    
    constraints = [sum_entries(x) == 1, x > 0]
    
    A = numpy.exp(trainll)
        
    objective = Maximize(sum_entries(log(A * x)))
    prob = Problem(objective, constraints)
    #prob.solve()
    # print("Optimal value", prob.solve())
    
    #ws = sum(x.value.tolist(), [])
    #print(ws/numpy.sum(ws))   
    return {'classes':classes, 'spns':spns, 'weights':ws}


def spnClassificationGeneralFit(X, Y, maxClasses, alpha=0.001, min_slices=500):
    # need to convert Y into one-hot encoding as there is no multinomial till now
    Y = getOneHotEncoding(Y, maxClasses)
    print('X shape : ',X.shape)
    print('Y shape : ',Y.shape)
    families = ['gaussian']*X.shape[1]+['binomial']*Y.shape[1]
    data_train_class = numpy.c_[X,Y]
    spn = LearnSPN(cache=memory, row_cluster_method="RandomPartition",ind_test_method="subsample",alpha=alpha, min_features_slice=10, min_instances_slice=min_slices, cluster_prep_method=None, families=families).fit_structure(data_train_class)
    return spn

def spnClassificationGeneralPred(model, X, maxClasses):
    print("Total number of instances : ",X.shape[0])
    predictions = numpy.zeros(X.shape[0])
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
        pmax = numpy.argmax(loglikelihood)
        predictions[i] = pmax
    return (predictions)

if __name__ == '__main__':
    '''
    input = numpy.loadtxt("data/w2v/in.txt").astype(int)
    output = numpy.loadtxt("data/w2v/out.txt").astype(int)
    '''
    features = numpy.loadtxt("data/w2v/w2v_embeddings.txt")
    classes = numpy.loadtxt("data/w2v/w2v_classes.txt").astype(int)
    classes = convert_to_contigous(classes)
    vocabsize = max(classes)+1
    #print('vocabsize : ',vocabsize)
    '''
    name = "twospirals"
    train1 = numpy.loadtxt("data/synthetic/"+name+".csv", delimiter=",")
    X = train1[:, (0, 1)]
    Y = train1[:, 2]
    '''
    X = features
    Y = classes
    k = 5 # Cross validations
    spn_shallow_shallow_accuracies = numpy.zeros(k) #stores accuracies of spn
    spn_shallow_deep_accuracies = numpy.zeros(k)
    spn_deep_shallow_accuracies = numpy.zeros(k)
    spn_deep_deep_accuracies = numpy.zeros(k)
    spn_deep_accuracies = numpy.zeros(k)
    spn_shallow_accuracies = numpy.zeros(k)
    max_prior_probs = numpy.zeros(k)
    logistic_probs = numpy.zeros(k)
    rfc_probs = numpy.zeros(k)
    nbc_probs = numpy.zeros(k)
    # Do cross validation
    for Xtrain, Ytrain, Xtest, Ytest, i in classkfolded(X, Y, 5):
        print("Fold number : ",i+1)
        
#         #Naive bayes classifier (multivariate gaussian features
#         nbc = naive_bayes.GaussianNB()
#         nbc.fit(Xtrain,Ytrain)
#         predictions = nbc.predict(Xtest)
#         nbc_probs[i] = (accuracy_score(Ytest, predictions))
#         #continue
#         
#         #Max prior predictions
#         val,counts = numpy.unique(Ytrain,return_counts = True)
#         prior_predictions = [numpy.argmax(counts)]*Ytest.shape[0]
#         max_prior_probs[i] = max(counts/sum(counts))
#         #continue
#         
#         #Logistic Regression
#         lrc = LogisticRegression()
#         lrc.fit(Xtrain,Ytrain)
#         predictions = lrc.predict(Xtest)
#         logistic_probs[i] = (accuracy_score(Ytest, predictions))
#         #continue
#     
#         #Random Forest
#         rfc = ensemble.RandomForestClassifier()
#         rfc.fit(Xtrain,Ytrain)
#         predictions = rfc.predict(Xtest)
#         rfc_probs[i] = (accuracy_score(Ytest, predictions))
#         #continue
#     
        result = open('result.txt','a')
        #Shallow SPN
        result.write("Learning shallow SPN...\n")
        start_time = time.time()
        spnmodel = spnClassificationGeneralFit(Xtrain,Ytrain,vocabsize,min_slices=Xtrain.shape[0])
        result.write("Learning SPN took %s seconds ---\n" % (time.time() - start_time))
        result.write("Testing shallow SPN...\n")
        start_time = time.time()
        spnPred = spnClassificationGeneralPred(spnmodel,Xtest,vocabsize)
        result.write("Testing SPN took %s seconds ---\n" % (time.time() - start_time))
        accuracy = accuracy_score(Ytest, spnPred)
        result.write("Acuracy : %.3f\n"%(accuracy))
        spn_shallow_accuracies[i] = accuracy
        result.close()
        
        
        # Deep SPN
        result = open('result.txt','a')
        result.write("Learning deep SPN...\n")
        start_time = time.time()
        spnmodel = spnClassificationGeneralFit(Xtrain,Ytrain,vocabsize)
        result.write("Learning SPN took %s seconds ---\n" % (time.time() - start_time))
        result.write("Testing Deep SPN...\n")
        start_time = time.time()
        spnPred = spnClassificationGeneralPred(spnmodel,Xtest,vocabsize)
        result.write("Testing SPN took %s seconds ---\n" % (time.time() - start_time))
        accuracy = accuracy_score(Ytest, spnPred)
        result.write("Acuracy : %.3f\n"%(accuracy))
        spn_deep_accuracies[i] = accuracy
        result.close()
        
        continue
        # Deep Shallow SPN
        print("Learning deep shallow SPN...")
        start_time = time.time()
        spnmodel = spnClassificationNBFit(Xtrain, Ytrain)
        print("Learning SPN took %s seconds ---" % (time.time() - start_time))
        print("Testing deep shallow SPN...")
        start_time = time.time()
        spnprediction, _ = spnClassificationNBPred(spnmodel, Xtest)
        print("Testing SPN took %s seconds ---" % (time.time() - start_time))
        spn_deep_shallow_accuracies[i] = accuracy_score(Ytest, spnprediction)
        
        # Deep Deep SPN
        print("Creating features for upper deep SPN...")
        start_time = time.time()
        _, trainll = spnClassificationNBPred(spnmodel, Xtrain)
        print("Features created in %s seconds ---" % (time.time() - start_time))
        Ytrain = getOneHotEncoding(Ytrain, vocabsize)
        print("Leraning upper deep SPN")
        start_time = time.time()
        spnmodel2 = spnClassificationGeneralFit((trainll), Ytrain)
        print("Learning SPN took %s seconds ---" % (time.time() - start_time))
        print("Testing deep deep SPN...creating features for upper SPN part")
        start_time = time.time()
        _, testll = spnClassificationNBPred(spnmodel, Xtest)
        print("Features created in %s seconds ---" % (time.time() - start_time))
        print("Testing deep deep SPN")
        start_time = time.time()
        testPred = spnClassificationGeneralPred(spnmodel2, testll, vocabsize)
        print("Testing SPN took %s seconds ---" % (time.time() - start_time))
        spn_deep_deep_accuracies[i] = accuracy_score(Ytest, testPred)
    
    print('nbc mean accuracy',numpy.mean(nbc_probs))
    print('maxprior mean accuracy',numpy.mean(max_prior_probs))
    print('logistic mean accuracy',numpy.mean(logistic_probs))
    print('Random Forest mean accuracy',numpy.mean(rfc_probs))
    print('Shallow spn accuracy',numpy.mean(spn_shallow_accuracies))
    print('Deep spn accuracy',numpy.mean(spn_deep_accuracies))
