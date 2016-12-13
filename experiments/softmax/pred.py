'''
Created on Nov 17, 2016

@author: molina
'''
from cvxpy import *
from joblib.memory import Memory
import numpy
from sklearn import datasets
from sklearn.feature_selection.tests.test_base import feature_names
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.classification import accuracy_score
from sklearn.model_selection._split import train_test_split

from algo.learnspn import LearnSPN
import matplotlib.pyplot as plt


memory = Memory(cachedir=".", verbose=0, compress=9)



def spnClassificationNBPred(model, X):
    
    trainll = numpy.zeros((X.shape[0],model['classes'].shape[0]))
    
    for j, spn in enumerate(model["spns"]):
        trainll[:, j] = spn.eval(X, individual=True) + numpy.log(model['weights'][j])

    pmax = numpy.argmax(trainll, axis=1)

    return model['classes'][pmax]



def spnClassificationNBFit(X, Y, alpha=0.001, min_slices=80):
    classes = numpy.unique(Y)
    spns = []
    print('classes : ',classes)
    trainll = numpy.zeros((X.shape[0],classes.shape[0]))
    ws = []
    print('shape : ',X.shape)
    for j in range(classes.shape[0]):
        idx = Y == classes[j]
        ws.append(float(numpy.sum(idx))/X.shape[0])
        
        data_train_class = X[idx, :]
        print('learning spn')
        families = 'gaussian'
        spn = LearnSPN(cache=None, alpha=alpha, min_instances_slice=min_slices, cluster_prep_method=None, families=families, cluster_first=False).fit_structure(data_train_class)
        spns.append(spn)
        print('spn learned')
        trainll[idx, j] = spn.eval(data_train_class, individual=True)
        

    x = Variable(len(classes))
    
    constraints = [sum_entries(x) == 1, x > 0]
    
    A = numpy.exp(trainll)
        
    objective = Maximize(sum_entries(log(A * x)))
    prob = Problem(objective, constraints)
    prob.solve()
    # print("Optimal value", prob.solve())
    
    #ws = sum(x.value.tolist(), [])
    print(ws)
        
    return {'classes':classes, 'spns':spns, 'weights':ws}





def spnClassificationSPNPred(model, X):
    
    trainll = numpy.zeros((X.shape[0],model['classes'].shape[0]))
    
    for j, spn in enumerate(model["spns"]):
        trainll[:, j] = spn.eval(X, individual=True) + numpy.log(model['weights'][j])

    pmax = numpy.argmax(trainll, axis=1)

    return model['classes'][pmax]



def spnClassificationSPNFit(X, Y, alpha=0.001, min_slices=80):
    classes = numpy.unique(Y)
    spns = []
    
    trainll = numpy.zeros((X.shape[0],classes.shape[0]))
    ws = []
    for j in range(classes.shape[0]):
        idx = Y == classes[j]
        ws.append(float(numpy.sum(idx))/X.shape[0])
        
        data_train_class = X[idx, :]
        spn = LearnSPN(cache=memory, alpha=alpha, min_instances_slice=min_slices, cluster_prep_method=None, families="gaussian").fit_structure(data_train_class)
        spns.append(spn)
        
        trainll[idx, j] = spn.eval(data_train_class, individual=True)
        

    x = Variable(len(classes))
    
    constraints = [sum_entries(x) == 1, x > 0]
    
    A = numpy.exp(trainll)
        
    objective = Maximize(sum_entries(log(A * x)))
    prob = Problem(objective, constraints)
    prob.solve()
    # print("Optimal value", prob.solve())
    
    #ws = sum(x.value.tolist(), [])
    print(ws)
        
    return {'classes':classes, 'spns':spns, 'weights':ws}

def spnClassificationGeneralFit(X, Y, maxClasses, alpha=0.001, min_slices=500, min_feature_slice = 30):
    # need to convert Y into one-hot encoding as there is no multinomial till now
    #Y = getOneHotEncoding(Y, maxClasses)
    print('X shape : ',X.shape)
    print('Y shape : ',Y.shape)
    families = ['gaussian']*X.shape[1]+['binomial']*Y.shape[1]
    data_train_class = numpy.c_[X,Y]
    spn = LearnSPN(cache=memory, row_cluster_method="RandomPartition",ind_test_method="subsample",alpha=alpha, min_features_slice=min_feature_slice, min_instances_slice=min_slices, cluster_prep_method=None, families=families).fit_structure(data_train_class)
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


def create_block_data(m,n):
    X,y = datasets.make_blobs(n_samples=m,n_features=n,centers=1)
    row1 = numpy.c_[X,numpy.zeros(X.shape)]
    row2 = numpy.c_[numpy.zeros(X.shape),X]
    block = numpy.r_[row1,row2]
    return block

if __name__ == '__main__':
#     data = create_block_data(5, 2)
#     print(data)
#     spn_model = spnClassificationGeneralFit(data, numpy.zeros((data.shape[0],1)), maxClasses=1, min_slices=2, min_feature_slice=1)
#     feature_names = ['x1','x2','x3','x4','x5']
#     spn_model.save_pdf_graph(featureNames = feature_names, outputfile='out_spn.pdf')
    numpy.random.seed(1222)
    X,y = datasets.make_blobs(1000,1,centers=2)
    train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.3)
    lrc = LogisticRegression()
    lrc.fit(X,y)
    pred_y = lrc.predict(test_x)
    print('coeff : ',lrc.coef_)
    print('intercept : %.3f'%(lrc.intercept_))
    print('accuracy : %.3f'%(accuracy_score(test_y,pred_y)))
    
    spnmodel = spnClassificationNBFit(train_x, train_y, min_slices=train_x.shape[0])
    spn_pred = spnClassificationNBPred(spnmodel, test_x)
    print('accuracy : %.3f'%(accuracy_score(test_y,spn_pred)))
    