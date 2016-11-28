from sklearn import tree,ensemble,svm,naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from scipy import stats

if __name__ == '__main__':
    features = np.loadtxt("data/w2v/w2v_embeddings.txt")
    classes = np.loadtxt("data/w2v/w2v_labels.txt").astype(int)
    nbc = naive_bayes.GaussianNB()
    nbc_scores = cross_val_score(nbc,features,classes,cv=5)
    print('Naive bayes mean accuracy : %.2f'%(nbc_scores.mean()))

    rfc = ensemble.RandomForestClassifier()
    rfc_scores = cross_val_score(rfc,features,classes,cv=5)
    print('Random forest mean accuracy : %.2f'%(rfc_scores.mean()))

    lrc = LogisticRegression()
    lrc_scores = cross_val_score(lrc,features,classes,cv=5)
    print('LR mean accuracy : %.2f'%(lrc_scores.mean()))

