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


memory = Memory(cachedir="/tmp", verbose=0, compress=9)


def spnClassificationRFPred(model, X):
    
    trainll = numpy.zeros((X.shape[0],model['classes'].shape[0]))
    
    for i, spn in enumerate(model["spns"]):
        j = i % model['classes'].shape[0]
        trainll[:, j] += numpy.exp(spn.eval(X, individual=True) + numpy.log(model['weights'][j]))

    pmax = numpy.argmax(trainll, axis=1)

    return model['classes'][pmax]
    

def spnClassificationRFFit(X, Y, random_spns = 10,alpha=0.001, min_slices=80):
    classes = numpy.unique(Y)
    spns = []
    
    ws = []
    instances_idx = []
    
    numpy.random.seed(1337)
    
    for i in range(random_spns):
        instances_idx.append(numpy.random.choice(X.shape[0], int(X.shape[0]*0.8) ))
        NX = X[instances_idx[-1],:]
        NY = Y[instances_idx[-1]]
        for j in range(classes.shape[0]):
            idx = NY == classes[j]
            ws.append(float(numpy.sum(idx))/NX.shape[0])
            
            data_train_class = NX[idx, :]
            spn = LearnSPN(cache=memory, alpha=alpha, min_instances_slice=min_slices, cluster_prep_method=None, families="gaussian").fit_structure(data_train_class)
            spns.append(spn)
            
            
    ws = ws/numpy.sum(ws)
    print(ws)
        
    return {'classes':classes, 'spns':spns, 'weights':ws, 'instances_idx':instances_idx}



def spnClassificationNBPred(model, X):
    
    trainll = numpy.zeros((X.shape[0],model['classes'].shape[0]))
    
    for j, spn in enumerate(model["spns"]):
        trainll[:, j] = spn.eval(X, individual=True) + numpy.log(model['weights'][j])

    pmax = numpy.argmax(trainll, axis=1)

    return model['classes'][pmax]



def spnClassificationNBFit(X, Y, alpha=0.001, min_slices=80):
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
    print(ws)
        
    return {'classes':classes, 'spns':spns, 'weights':ws}

if __name__ == '__main__':
    name = "twospirals"
    train1 = numpy.loadtxt("data/synthetic/"+name+".csv", delimiter=",")
    
    print(train1.shape)
    print(numpy.bincount(train1[:,2].astype(numpy.uint32)))
    #0/0
    
#     test1 = train1 #numpy.loadtxt("data/synthetic/clusterincluster.csv")
#     
#     X = train1[:, (0, 1)]
#     Y = train1[:, 2]
#     model = spnClassificationNBFit(X, Y)
#     #model = spnClassificationRFFit(X, Y)
#     
#     
#     
#     XT = test1[:, (0, 1)]
#     YT = test1[:, 2]
#     prediction = spnClassificationNBPred(model, XT)
#     #prediction = spnClassificationRFPred(model, XT)
#     print(accuracy_score(YT, prediction))
#     
#     
#     xvals = numpy.arange(numpy.min(X[:,0])-3, numpy.max(X[:,0])+3, 0.1)
#     yvals = numpy.arange(numpy.min(X[:,1])-3, numpy.max(X[:,1])+3, 0.1)
#     
#     XT2 = numpy.zeros((xvals.shape[0]*yvals.shape[0],2))
#     i = 0
#     for xv in xvals:
#         for yv in yvals:
#             XT2[i,:] = [xv,yv]
#             i += 1
#     
#     Y2 = spnClassificationNBPred(model, XT2)
#     #Y2 = spnClassificationRFPred(model, XT2)
#     
#     
#     sym = ["r.", "b.", "g.", "k."]
# 
# 
#     for c in numpy.unique(Y2).astype(int):
#         plt.plot(XT2[Y2==c,0], XT2[Y2==c,1], sym[c], markersize=2)
#     
#     sym = ["ro", "bo", "go", "ko"]    
#     for c in numpy.unique(Y).astype(int):
#         plt.plot(X[Y==c,0], X[Y==c,1], sym[c])
#         pass
#     plt.savefig("data/synthetic/"+name+'.png')

        
   
   