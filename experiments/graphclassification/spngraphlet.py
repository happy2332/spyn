
from collections import OrderedDict
import glob
import json
import os
from pprint import pprint
from statistics import mean, stdev

from joblib.memory import Memory
import numpy
from sklearn import preprocessing
from sklearn import svm, cross_validation
from sklearn.cross_validation import train_test_split, StratifiedKFold, KFold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, metrics
from sklearn.metrics import classification_report
from sklearn.metrics.ranking import roc_curve, auc
from statsmodels.genmod.cov_struct import Independence

from algo.learnspn import LearnSPN
from pdn.independenceptest import getPval

import time

memory = Memory(cachedir="/data/d1/molina/spn", verbose=0, compress=9)

path = os.path.dirname(__file__)

    
def evalModel(predictor, train_data, train_labels, test_data, test_labels, name, evalresults):

    
    predictor.fit(train_data, train_labels)
    evalresults.setdefault(name + " Accuracy raw \t\t", []).append(metrics.accuracy_score(test_labels, predictor.predict(test_data)))

    predictor.fit(preprocessing.scale(train_data), train_labels)
    evalresults.setdefault(name + " Accuracy std \t\t", []).append(metrics.accuracy_score(test_labels, predictor.predict(preprocessing.scale(test_data))))
    
    predictor.fit(preprocessing.normalize(train_data, norm='l2'), train_labels)
    evalresults.setdefault(name + " Accuracy nml \t\t", []).append(metrics.accuracy_score(test_labels, predictor.predict(preprocessing.normalize(test_data, norm='l2'))))

    if len(set(train_labels)) != 2:
        return
    
    predictor.fit(train_data, train_labels)
    fpr, tpr, _ = roc_curve(test_labels, predictor.decision_function(test_data))
    evalresults.setdefault(name + " AUC raw \t\t", []).append(auc(fpr, tpr))

    predictor.fit(preprocessing.scale(train_data), train_labels)
    fpr, tpr, _ = roc_curve(test_labels, predictor.decision_function(preprocessing.scale(test_data)))
    evalresults.setdefault(name + " AUC std \t\t", []).append(auc(fpr, tpr))
    
    
    predictor.fit(preprocessing.normalize(train_data, norm='l2'), train_labels)
    fpr, tpr, _ = roc_curve(test_labels, predictor.decision_function(preprocessing.normalize(test_data, norm='l2')))
    evalresults.setdefault(name + " AUC nml \t\t", []).append(auc(fpr, tpr))



#def evalspnCVFold(labels, data, train_index, test_index):


def evalspnComplete(labels, data, dsname, writer, alpha, min_slices=50):
    
    cvfolds = StratifiedKFold(labels, n_folds=5, random_state=123)
    classes = list(set(labels))
    
    evalresults = OrderedDict()
    
    for train_index, test_index in cvfolds:
        train_data = data[train_index, ]
        train_labels = labels[train_index]
        
        test_data = data[test_index, ]
        test_labels = labels[test_index]
        
        clfsvc = GridSearchCV(estimator=svm.SVC(kernel='linear', probability=True), param_grid=dict(C=numpy.logspace(-10, 0, 10)), n_jobs=50, cv=5)
        clflr = LogisticRegression(solver='lbfgs')
        
        start = time.time()
        evalModel(clfsvc, test_data, test_labels, train_data, train_labels, "SVM raw", evalresults)
        evalresults.setdefault("SVM time in secs \t\t", []).append((time.time() - start))
        
        evals_train = numpy.zeros((train_data.shape[0], 0))
        evals_test = numpy.zeros((test_data.shape[0], 0))

        grads_train = numpy.zeros((train_data.shape[0], 0))
        grads_test = numpy.zeros((test_data.shape[0], 0))
        
        activations_train = numpy.zeros((train_data.shape[0], 0))
        activations_test = numpy.zeros((test_data.shape[0], 0))
        
        timespn = 0
        for c in classes:
            idx = train_labels == c
            data_train_class = train_data[idx, :]
            
            start = time.time()
            spn = LearnSPN(alpha=alpha, min_instances_slice=min_slices, cluster_prep_method="sqrt", cache=memory).fit_structure(data_train_class)
            #spn = spnlearn(data_train_class, alpha, min_slices=min_slices, cluster_prep_method="sqrt", family="poisson")
            timespn += (time.time() - start)
            
            #continue
            evalperclass = numpy.asarray(spn.eval(train_data, individual=True)).reshape((train_data.shape[0],1))
            print(evalperclass.shape)
            print(evalperclass)
            gradsperclass = spn.gradients(train_data)
            activationperclass = spn.activations(train_data)
            print(evals_train.shape)
            evals_train = numpy.append(evals_train, evalperclass, axis=1)
            print(evals_train)
            grads_train = numpy.hstack((grads_train, gradsperclass))
            activations_train = numpy.hstack((activations_train, activationperclass))
            
            evals_test = numpy.hstack((evals_test, numpy.asarray(spn.eval(test_data, individual=True)).reshape((test_data.shape[0],1))))
            grads_test = numpy.hstack((grads_test, spn.gradients(test_data)))
            activations_test = numpy.hstack((activations_test, spn.activations(test_data)))
            print("loop done")
            
        evalresults.setdefault("SPN time in secs \t\t", []).append(timespn)
         
        evalModel(clflr, evals_test, test_labels, evals_train, train_labels, "SPN per class ll -> LR", evalresults)
    
        evalModel(clfsvc, grads_test, test_labels, grads_train, train_labels, "SPN per class gradients -> SVM", evalresults)
        
        evalModel(clfsvc, activations_test, test_labels, activations_train, train_labels, "SPN per class activations -> SVM", evalresults)
    
    
    writer.write(json.dumps(evalresults))
    writer.write("\n")
    
    
    for key, value in evalresults.items():
        writer.write("%s: %0.6f (+/- %0.6f) \n" % (key, mean(value), stdev(value) * 2))
        
    writer.write("\n")

def run(classfname, datafname):
    
    classlabels = numpy.loadtxt(classfname)
    
    data = numpy.loadtxt(datafname, dtype=int, delimiter=",")
    alpha=0.001
    print(data.shape)
    print(classlabels.shape)
    
    with open("output1.txt", "a", buffering=2) as myfile:
        myfile.write(classfname + "\n")
        myfile.write(datafname + "\n")
        myfile.write('%.30g'%(alpha) + "\n")        
        evalspnComplete(classlabels, data, os.path.basename(datafname), myfile, alpha, min_slices=80)
    
if __name__ == '__main__':
    
    
    for fname in sorted(glob.glob('wl/1*corpus.classes.csv'), reverse=True):
        classfname = fname
        datafname = fname.replace(".classes", "")
        print("running for %s " % (datafname))
        print("running for %s " % (classfname))
        
        run(classfname, datafname)
        
    
