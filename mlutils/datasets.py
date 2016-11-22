import codecs
import datetime
import os
import re

from joblib import memory
from numpy import float64
import numpy


# from rpy2 import robjects
# from rpy2.robjects import numpy2ri
# from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage
path = os.path.dirname(__file__) + "/datasets/"


# with open (path+"/pgen.R", "r") as pdnfile:
    # pgen = SignatureTranslatedAnonymousPackage(''.join(pdnfile.readlines()), "pgen")
#    pass

# @memory.cache
# def getpdata(n,p,type="hub"):
#     numpy2ri.activate()
#     try:
#         #out = pgen.pgen(n,p,type)
#         out = None
#     except Exception as e:
#         print(e)
#         raise e 
#         
#     pdata = numpy.array(out)
#     numpy2ri.deactivate()
#     return pdata

def getTraffic():
    fname = path + "/traffic.csv"
    words = str(open(fname, "rb").readline()).split(';')
    
    words = list(map(lambda w: w.replace('"', '').strip(), words))
    words = words[2:]
    
    D = numpy.loadtxt(fname, dtype="S20", delimiter=";", skiprows=1)
    times = D[:, 1]
    D = D[:, 2:]
    
    
    nas = numpy.zeros_like(D, dtype="S20")
    nas[:] = "NA"
    
    times = times[numpy.all(D != nas, axis=1)]
    D = D[numpy.all(D != nas, axis=1)]
    
    D = D.astype(float)
    
    hours = map(lambda t: float(datetime.datetime.fromtimestamp(
        int(t)
    ).strftime('%H')), times)

    
    hours = numpy.asmatrix(hours).T
        
    return (D, words, times, hours)


def getGrolier():
    
    words = list(map(lambda line: line.decode(encoding='UTF-8').strip(), open(path + "grolier15276_words.txt", "rb").readlines()))
    
    documents = list(map(lambda line: line.decode(encoding='UTF-8').strip().replace(",,", ""), open(path + "grolier15276.csv", "rb").readlines()))
    
    D = numpy.zeros((len(documents), len(words)))
    
    for i, doc in enumerate(documents):
        doc = doc.split(",")[1:]
        for j in range(0, len(doc), 2):
            D[i, int(doc[j]) - 1] = int(doc[j + 1])
            
    return ("Grolier", D, words)

def getNips():
    fname = path + "nips100.csv"
    words = open(fname, "rb").readline().decode(encoding='UTF-8').strip().split(',')
    D = numpy.loadtxt(fname, dtype=float, delimiter=",", skiprows=1)
    return ("Nips", D, words)

def getMSNBCclicks():
    fname = path + "MSNBC.pdn.csv"
    words = open(fname, "rb").readline().decode(encoding='UTF-8').strip().split(',')
    D = numpy.loadtxt(fname, dtype=float, delimiter=",", skiprows=1)
    return ("MSNBC", D, words)



def getSynthetic():
    fname = path + "synthetic.csv"
    words = sum([["A" + str(i) for i in range(1, 51)], ["B" + str(i) for i in range(1, 51)]], [])
    
    D = numpy.loadtxt(fname, dtype=float, delimiter=",", skiprows=0)
    return ("Synthetic", D, words)

def getCommunitiesAndCrimes(filtered=True):
    words = map(lambda x: x.decode(encoding='UTF-8').strip(), open(path + "communities_names.txt", "rb").readlines())
    words = [word.split() for word in words if word.startswith("@attribute")]
    
    D = numpy.loadtxt(path + "communities.txt", dtype='S8', delimiter=",").view(numpy.chararray).decode('utf-8')
    
    if not filtered:
        return ("C&C", D, words)
    
    
    numidx = [i for i in range(len(words)) if words[i][2] == "numeric"]
    words = [words[i][1] for i in range(len(words)) if words[i][2] == "numeric"]
    
    D = D[:, numidx]
    
    words = [words[-18], words[-16], words[-14], words[-12], words[-10], words[-8], words[-6], words[-4]]
    D = D[:, (-18, -16, -14, -12, -10, -8, -6, -4)]
    
    denseidx = [r for r in range(D.shape[0]) if not any(D[r, :] == "?")]
    D = D[denseidx, :]
    
    D = D.astype(float)
    return ("C&C", D, words)


def getSpambase(instances=4601):
    words = map(lambda x: x.decode(encoding='UTF-8').strip(), open(path + "spambase_names.txt", "rb").readlines())
    words = [word.split() for word in words if word.startswith("@attribute")]

    D = numpy.loadtxt(path + "spambase.txt", dtype='S8', delimiter=",").view(numpy.chararray).decode('utf-8')

    numidx = [i for i in range(len(words)) if words[i][2] == "REAL"]
    words = [words[i][1] for i in range(len(words)) if words[i][2] == "REAL"]
    numidx.append(57) # class column
    words.append('class') # class column

    D = D[:instances, numidx]
    #D = D[1000:3000, numidx]
    D = D.astype(float)
    return ("Spambase", D, words)


def getIRIS(classes=2):
    assert classes == 2 or classes == 3
    from sklearn import datasets
    from sklearn.cross_validation import train_test_split

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    labels = iris.feature_names
    y = y.reshape(len(y), 1)
    data = numpy.hstack((y,X))
    data = data[:(classes*50), :]  # 50 instances per class

    labels = numpy.append('class', labels)

    families = ['gaussian' for i in range(len(labels))]
    families[0] = 'binomial'  # first column is class label

    data = data.astype(float)
    numpy.random.seed(42)
    numpy.random.shuffle(data)
    return "IRIS"+str(classes), data, labels, classes, families

def getSyntheticClassification(classes, features, informative, samples):
    import sklearn.datasets
    X, y = sklearn.datasets.make_classification(n_samples=samples, n_features=features, n_informative=informative, n_classes=classes, random_state=42, shuffle= True)
    y = y.reshape(len(y), 1)
    data = numpy.hstack((y, X))
    labels = ['x' + str(i) for i in range(features)]
    labels.insert(0, 'class')

    families = ['gaussian' for i in range(len(labels))]
    families[0] = 'binomial'  # first column is class label

    name = "SYN_" + str(classes) + '_' + str(features) + '_' + str(samples)

    return name, data, labels, classes, families


def getWisconsinBreastCancer():

    fname = path + "wisonsinbreastcancer.csv"

    labels = open(fname, "rb").readline().decode(encoding='UTF-8').strip().split(',')
    labels = list(map(lambda w: w.replace('"', '').strip(), labels))
    data = numpy.loadtxt(fname, dtype=float, delimiter=",", skiprows=1)
    data = data.astype(float)

    # remove "id" column
    labels.remove('id')
    data = numpy.delete(data, 0, axis=1)

    families = ['gaussian' for i in range(len(labels))]
    families[0] = 'binomial'  # first column is class label
    return "BreastCancer", data, labels, 2, families


def getGlass():

    fname = path + "glass.csv"

    labels = open(fname, "rb").readline().decode(encoding='UTF-8').strip().split(',')
    labels = list(map(lambda w: w.replace('"', '').strip(), labels))

    data = numpy.loadtxt(fname, dtype=float, delimiter=",", skiprows=1)
    data = data.astype(float)

    # remove "id" column
    labels.remove('ID')
    data = numpy.delete(data, 0, axis=1)

    # change class index to first column
    labels = numpy.roll(labels, 1)
    data = numpy.roll(data, 1, axis=1)

    families = ['gaussian' for i in range(len(labels))]
    families[0] = 'binomial'  # first column is class label

    # TODO refactor classes starting with 0
    return "Glass", data, labels, 6, families


def getDiabetes():

    fname = path + "pima-indians-diabetes.data.csv"

    labels = open(fname, "rb").readline().decode(encoding='UTF-8').strip().split(',')
    labels = list(map(lambda w: w.replace('"', '').strip(), labels))

    data = numpy.loadtxt(fname, dtype=float, delimiter=",", skiprows=1)
    data = data.astype(float)

    # change class index to first column
    labels = numpy.roll(labels, 1)
    data = numpy.roll(data, 1, axis=1)

    families = ['gaussian' for i in range(len(labels))]
    families[0] = 'binomial'  # first column is class label
    return "Diabetes", data, labels, 2, families


def getIonosphere():
    fname = path + "ionosphere.data.csv"

    # bad = class 0
    # good = class 1
    data = numpy.loadtxt(fname, dtype=float, delimiter=",", skiprows=0)
    data = data.astype(float)

    labels = ['x'+str(i) for i in range(data.shape[1])]
    labels[data.shape[1]-1] = 'class'

    # remove x2 - always 0
    labels.remove('x1')
    data = numpy.delete(data, 1, axis=1)

    labels.remove('x0')
    data = numpy.delete(data, 0, axis=1)

    # change class index to first column
    labels = numpy.roll(labels, 1)
    data = numpy.roll(data, 1, axis=1)

    families = ['gaussian' for i in range(len(labels))]
    families[0] = 'binomial'  # first column is class label
    return "Ionosphere", data, labels, 2, families


def getWineQualityWhite():

    fname = path + "winequality-white.csv"

    labels = open(fname, "rb").readline().decode(encoding='UTF-8').strip().split(';')
    labels = list(map(lambda w: w.replace('"', '').strip(), labels))
    data = numpy.loadtxt(fname, dtype=float, delimiter=";", skiprows=1)
    data = data.astype(float)

    #change index to first column
    labels = numpy.roll(labels, 1)
    data = numpy.roll(data, 1, axis=1)

    #make three classes:
    print(len(numpy.where(data[:, 0] <= 2)[0]))
    print(len(numpy.where((data[:, 0] <= 5))[0]))
    print(len(numpy.where((data[:, 0] >= 6) & (data[:, 0] <= 7))[0]))
    print(len(numpy.where((data[:, 0] == 7))[0]))


    return "WineQuality", data, labels

def removeOutliers(dd, deviations=5):
    (dsname, data, featureNames) = dd
    print(list(range(data.shape[1])))
    for col in range(data.shape[1]):
        colsdata = data[:,col]
        dataidx = abs(colsdata - numpy.median(colsdata)) < deviations * numpy.std(colsdata)
        data = data[dataidx, ]
    
    return (dsname, data, featureNames)

if __name__ == '__main__':
    print(getCommunitiesAndCrimes()[1].shape)
    
    print(removeOutliers(getCommunitiesAndCrimes())[1].shape)
    #print(len(getNips()[2]))
    # print(getCommunitiesAndCrimes())
    # print(getSynthetic())
    # print(getMSNBCclicks())
    #print(getGrolier())
