'''
Created on 12.06.2016

@author: alejomc
'''
from sklearn.cross_validation import KFold
from numpy import asarray


def kfolded(data, folds):
    kf = KFold(data.shape[0], n_folds=folds)
    
    for i, (train_index, test_index) in enumerate(kf):
        if len(data.shape) > 1:
            yield (data[train_index,:], data[test_index,:], i)
        else:
            yield (data[train_index], data[test_index], i)


if __name__ == '__main__':

    for train, test, i in kfolded(asarray(range(10)), 4):
        print(train, test, i)