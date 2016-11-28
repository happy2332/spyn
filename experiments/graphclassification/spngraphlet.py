
from collections import OrderedDict
from cvxpy import *
import glob
from joblib.memory import Memory
from joblib.test.test_pool import check_array
import json
import numpy
import os
from pprint import pprint
from sklearn import preprocessing
from sklearn import svm, cross_validation
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_validation import train_test_split, StratifiedKFold, KFold
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics.ranking import roc_curve, auc
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_is_fitted
from statistics import mean, stdev
from statsmodels.genmod.cov_struct import Independence
import time

from algo.learnspn import LearnSPN
from pdn.independenceptest import getPval


memory = Memory(cachedir="/tmp/spn3", verbose=0, compress=9)

path = os.path.dirname(__file__)


class SPNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, alpha=0.001, min_instances_slice=80, families="poisson"):
        self.alpha = alpha
        self.min_instances_slice = min_instances_slice
        self.families = families


    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"alpha": self.alpha, "min_instances_slice": self.min_instances_slice, "families": self.families}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        print(y.shape, numpy.unique(y))
        print(self.classes_)
        
        #0/0

        self.X_ = X
        self.y_ = y
        # Return the classifier
        
        
        # classes = numpy.unique(Y)
        self.spns_ = []
        
        self.ws_ = []
        trainll = numpy.zeros((X.shape[0],self.classes_.shape[0]))
        for j in range(self.classes_.shape[0]):
            idx = y == self.classes_[j]
            #self.ws_.append(float(numpy.sum(idx)) / X.shape[0])
            
            data_train_class = X[idx, :]
            spn = LearnSPN(alpha=self.alpha, min_instances_slice=self.min_instances_slice, cluster_prep_method="sqrt", families=self.families, cache=memory).fit_structure(data_train_class)
            self.spns_.append(spn)
            trainll[idx, j] = spn.eval(data_train_class, individual=True)
        
        
        #self.ws_ = self.ws_/numpy.sum(self.ws_)
        
        
        x = Variable(self.classes_.shape[0])
    
        constraints = [sum_entries(x) == 1, x > 0]
        
        A = numpy.exp(trainll)
            
        objective = Maximize(sum_entries(log(A * x)))
        prob = Problem(objective, constraints)
        prob.solve()
        
        
        self.ws_ = sum(x.value.tolist(), [])
        #print("Optimal w",self.ws_)
        
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        #X = check_array(X)
        
        print(X.shape)
        
        if not isinstance(X, numpy.ndarray):
            raise RuntimeError("input not an array")
        
        if X.shape[1] != self.X_.shape[1]:
            raise RuntimeError("invalid number of dimensions")
        
        #if numpy.any(X < 0):
        #    raise RuntimeError("invalid domain")
                
        if numpy.any(numpy.isnan(X)):
            raise RuntimeError("invalid type")
        

        trainll = numpy.zeros((X.shape[0], self.classes_.shape[0]))
    
        for j, spn in range(self.classes_.shape[0]):
            trainll[:, j] = spn.eval(X, individual=True) + numpy.log(self.ws_[j])
    
        #return self.clflr_.predict(trainll)
    
        pmax = numpy.argmax(trainll, axis=1)
    
        return self.classes_[pmax], trainll

    
#check_estimator(SPNClassifier)
    
def evalModel(predictor, test_data, test_labels, train_data, train_labels, name, evalresults):

    
    predictor.fit(train_data, train_labels)
    evalresults.setdefault(name + " Accuracy raw \t\t", []).append(accuracy_score(test_labels, predictor.predict(test_data)))

    #predictor.fit(preprocessing.scale(train_data), train_labels)
    #evalresults.setdefault(name + " Accuracy std \t\t", []).append(metrics.accuracy_score(test_labels, predictor.predict(preprocessing.scale(test_data))))
 
    #predictor.fit(preprocessing.normalize(train_data, norm='l2'), train_labels)
    #evalresults.setdefault(name + " Accuracy nml \t\t", []).append(metrics.accuracy_score(test_labels, predictor.predict(preprocessing.normalize(test_data, norm='l2'))))

    return

    if len(set(train_labels)) != 2:
        return
    
    predictor.fit(train_data, train_labels)
    fpr, tpr, _ = roc_curve(test_labels, predictor.decision_function(test_data))
    evalresults.setdefault(name + " AUC raw \t\t", []).append(auc(fpr, tpr))

    #return

    predictor.fit(preprocessing.scale(train_data), train_labels)
    fpr, tpr, _ = roc_curve(test_labels, predictor.decision_function(preprocessing.scale(test_data)))
    evalresults.setdefault(name + " AUC std \t\t", []).append(auc(fpr, tpr))
    
    
    predictor.fit(preprocessing.normalize(train_data, norm='l2'), train_labels)
    fpr, tpr, _ = roc_curve(test_labels, predictor.decision_function(preprocessing.normalize(test_data, norm='l2')))
    evalresults.setdefault(name + " AUC nml \t\t", []).append(auc(fpr, tpr))



# def evalspnCVFold(labels, data, train_index, test_index):


def evalspnComplete(labels, data, dsname, writer, alpha, min_instances_slice=50):
    
    cvfolds = StratifiedKFold(labels, n_folds=10, random_state=123)
    classes = list(set(labels))
    
    evalresults = OrderedDict()
    
    for train_index, test_index in cvfolds:
        train_data = data[train_index, ]
        train_labels = labels[train_index]
        
        test_data = data[test_index, ]
        test_labels = labels[test_index]
        
        # clfsvc = GridSearchCV(estimator=svm.SVC(kernel='linear', probability=True), param_grid=dict(C=numpy.logspace(-10, 0, 10)), n_jobs=50, cv=5)
        clfsvc = GridSearchCV(estimator=svm.SVC(kernel='linear', probability=True), param_grid={'C': [10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3]}, n_jobs=50, cv=5)
        start = time.time()
        evalModel(clfsvc, test_data, test_labels, train_data, train_labels, "SVM raw", evalresults)
        evalresults.setdefault("SVM time in secs \t\t", []).append((time.time() - start))
        
        clspn = SPNClassifier(alpha=alpha, min_instances_slice=min_instances_slice)
        start = time.time()
        evalModel(clspn, test_data, test_labels, train_data, train_labels, "SPN NB raw", evalresults)
        evalresults.setdefault("SPN time in secs \t\t", []).append((time.time() - start))
        
        #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        #clflr = LogisticRegression(solver='lbfgs')
        #start = time.time()
        #evalModel(clflr, test_data, test_labels, train_data, train_labels, "LR NB raw", evalresults)
        #evalresults.setdefault("SPN time in secs \t\t", []).append((time.time() - start))
        continue
        
        evals_train = numpy.zeros((train_data.shape[0], 0))
        evals_test = numpy.zeros((test_data.shape[0], 0))

        grads_train = numpy.zeros((train_data.shape[0], 0))
        grads_test = numpy.zeros((test_data.shape[0], 0))
        
        activations_train = numpy.zeros((train_data.shape[0], 0))
        activations_test = numpy.zeros((test_data.shape[0], 0))
        
        
        #model = ClassificationNBFit(train_data, train_labels)
        
        timespn = 0
        for c in classes:
            #break
            idx = train_labels == c
            print(idx)
            data_train_class = train_data[idx, :]
            
            start = time.time()
            spn = LearnSPN(alpha=alpha, min_instances_slice=min_instances_slice, cluster_prep_method="sqrt", cache=memory).fit_structure(data_train_class)
            print(alpha, min_instances_slice)
            # spn = spnlearn(data_train_class, alpha, min_slices=min_slices, cluster_prep_method="sqrt", family="poisson")
            timespn += (time.time() - start)
            
            # continue
            evalperclass = numpy.asarray(spn.eval(train_data, individual=True)).reshape((train_data.shape[0], 1))
            print(evalperclass.shape)
            print(evalperclass)
            gradsperclass = spn.gradients(train_data)
            activationperclass = spn.activations(train_data)
            print(evals_train.shape)
            evals_train = numpy.append(evals_train, evalperclass, axis=1)
            print(evals_train)
            grads_train = numpy.hstack((grads_train, gradsperclass))
            activations_train = numpy.hstack((activations_train, activationperclass))
            
            evals_test = numpy.hstack((evals_test, numpy.asarray(spn.eval(test_data, individual=True)).reshape((test_data.shape[0], 1))))
            grads_test = numpy.hstack((grads_test, spn.gradients(test_data)))
            activations_test = numpy.hstack((activations_test, spn.activations(test_data)))
            print("loop done")
            
        evalresults.setdefault("SPN time in secs \t\t", []).append(timespn)
         
        
        
        evalModel(clflr, evals_test, test_labels, evals_train, train_labels, "SPN per class ll -> LR", evalresults)
    
        #evalModel(clfsvc, grads_test, test_labels, grads_train, train_labels, "SPN per class gradients -> SVM", evalresults)
        
        #evalModel(clfsvc, activations_test, test_labels, activations_train, train_labels, "SPN per class activations -> SVM", evalresults)
    
    
    writer.write(json.dumps(evalresults))
    writer.write("\n")
    
    
    for key, value in evalresults.items():
        writer.write("%s: %0.6f (+/- %0.6f) \n" % (key, mean(value), stdev(value) * 2))
        
    writer.write("\n")

def run(classfname, datafname):
    
    classlabels = numpy.loadtxt(classfname)
    
    data = numpy.loadtxt(datafname, dtype=int, delimiter=",")
    
    alpha = 0.001
    print(data.shape)
    print(classlabels.shape)
    
    with open("output5.txt", "a", buffering=2) as myfile:
        myfile.write(classfname + "\n")
        myfile.write(datafname + "\n")
        myfile.write('%.30g' % (alpha) + "\n")        
        evalspnComplete(classlabels, data, os.path.basename(datafname), myfile, alpha, min_instances_slice=80)
    
if __name__ == '__main__':
    
    
    for fname in sorted(glob.glob('wl/2*corpus.classes.csv'), reverse=True):
        classfname = fname
        datafname = fname.replace(".classes", "")
        print("running for %s " % (datafname))
        print("running for %s " % (classfname))
        
        run(classfname, datafname)
        
    
