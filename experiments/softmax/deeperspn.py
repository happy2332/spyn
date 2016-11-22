'''
Created on Nov 17, 2016

@author: molina
'''
from cvxpy import *
from joblib.memory import Memory
import numpy
from sklearn.metrics.classification import accuracy_score
import matplotlib.pyplot as plt
from algo.learnspn import LearnSPN

numpy.set_printoptions(threshold=numpy.inf)

memory = Memory(cachedir="/tmp", verbose=0, compress=9)

def getOneHotEncoding(data, vocab_size=100):
    result = numpy.zeros((data.shape[0], vocab_size))
    
    result[list(range(data.shape[0])),data] = 1
    return result


def spnClassificationNBPred(model, X):
    
    trainll = numpy.zeros((X.shape[0],model['classes'].shape[0]))
    
    for j, spn in enumerate(model["spns"]):
        trainll[:, j] = spn.eval(X, individual=True) + numpy.log(model['weights'][j])

    pmax = numpy.argmax(trainll, axis=1)

    return (model['classes'][pmax], trainll)



def spnClassificationNBFit(X, Y, alpha=0.001, min_slices=80):
    classes = numpy.unique(Y)
    spns = []
    
    trainll = numpy.zeros((X.shape[0],classes.shape[0]))
    ws = []
    for j in range(classes.shape[0]):
        idx = Y == classes[j]
        ws.append(float(numpy.sum(idx))/X.shape[0])
        
        data_train_class = X[idx, :]
        spn = LearnSPN(cache=memory, alpha=alpha, min_instances_slice=min_slices, cluster_prep_method=None, family="gaussian").fit_structure(data_train_class)
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
    print(ws/numpy.sum(ws))
        
    return {'classes':classes, 'spns':spns, 'weights':ws}

if __name__ == '__main__':
    input = numpy.loadtxt("data/w2v/in.txt").astype(int)
    output = numpy.loadtxt("data/w2v/out.txt").astype(int)
    
    datain = getOneHotEncoding(input, 100)
    #dataout = getOneHotEncoding(output, 100)
    
    X = datain
    Y = output
    
    model = spnClassificationNBFit(X, Y)
    prediction, trainll = spnClassificationNBPred(model, X)
    print(X.shape,100)
    print(accuracy_score(Y, prediction))
    
    model = spnClassificationNBFit(X, Y, min_slices=1000000)
    prediction, trainll = spnClassificationNBPred(model, X)
    print(X.shape,100)
    print(accuracy_score(Y, prediction))
    #print(input)
    
    #print(datain, datain.shape)
    0/0


if __name__ == '__main__':
    name = "twospirals"
    train1 = numpy.loadtxt("data/synthetic/"+name+".csv", delimiter=",")
    
    print(train1)
    print(train1[:,2])
    #0/0
    
    test1 = train1 #numpy.loadtxt("data/synthetic/clusterincluster.csv")
    
    X = train1[:, (0, 1)]
    Y = train1[:, 2]
    #model = spnClassificationNBFit(X, Y)
    model = spnClassificationRFFit(X, Y)
    
    
    
    XT = test1[:, (0, 1)]
    YT = test1[:, 2]
    #prediction = spnClassificationNBPred(model, XT)
    prediction = spnClassificationRFPred(model, XT)
    print(accuracy_score(YT, prediction))
    
    
    xvals = numpy.arange(numpy.min(X[:,0])-3, numpy.max(X[:,0])+3, 0.1)
    yvals = numpy.arange(numpy.min(X[:,1])-3, numpy.max(X[:,1])+3, 0.1)
    
    XT2 = numpy.zeros((xvals.shape[0]*yvals.shape[0],2))
    i = 0
    for xv in xvals:
        for yv in yvals:
            XT2[i,:] = [xv,yv]
            i += 1
    
    #Y2 = spnClassificationNBPred(model, XT2)
    Y2 = spnClassificationRFPred(model, XT2)
    
    
    sym = ["r.", "b.", "g.", "k."]


    for c in numpy.unique(Y2).astype(int):
        plt.plot(XT2[Y2==c,0], XT2[Y2==c,1], sym[c], markersize=2)
    
    sym = ["ro", "bo", "go", "ko"]    
    for c in numpy.unique(Y).astype(int):
        plt.plot(X[Y==c,0], X[Y==c,1], sym[c])
        pass
    plt.savefig("data/synthetic/"+name+'.png')

        
   
   