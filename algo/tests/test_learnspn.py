import algo.learnspn
from algo.dataslice import DataSlice

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import math

import random

import numpy
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

from spn.linked.nodes import ProductNode
from spn.linked.nodes import SumNode

import dataset

import logging


seed = 1337
numpy.random.seed(seed)
rand_gen = numpy.random.RandomState(seed)

n_rows = 10
n_cols = 5
#
# generating test data randomly
data = numpy.random.binomial(n=1, p=0.5, size=(n_rows, n_cols))
print('syntetich data', data)

#
# assuming all features to be binary r.v.s
feature_vals = [2 for i in range(n_cols)]





def read_ivals_from_file(dataset_name,
                         sep=' ',
                         extension='.ts.data_ival',
                         path='algo/tests/data/',
                         verbose=False):
    #
    # composing the full path
    filename = path + dataset_name + extension
    print('Opening the file', filename)

    #
    # reading all lines at once
    with open(filename) as file_p:
        all_lines = file_p.readlines()

    #
    # parsing each line

    i_vals = [[int(i_val) for i_val in line.rstrip().split(sep)]
              for line in all_lines]
    #
    # printing can be expensive
    if verbose:
        for i_val_line in i_vals:
            print(i_val_line)

    return i_vals



def test_retrieve_clustering():
    assignment = [2, 3, 8, 3, 1]
    clustering = algo.learnspn.retrieve_clustering(assignment)
    print(clustering)
    #
    # this test is much stricter than needed
    assert clustering == [[0], [1, 3], [2], [4]]
    #
    # checking even when indexes are provided
    indexes = [21, 1, 4, 18, 11]
    clustering = algo.learnspn.retrieve_clustering(assignment, indexes)
    print(clustering)
    assert clustering == [[21], [1, 18], [4], [11]]


def test_cluster_rows_GMM():

    #
    # random generator
    seed = 1337
    rand_gen = numpy.random.RandomState(seed)

    #
    # loading a very simple dataset
    dataset_name = 'nltcs'
    train, valid, test = dataset.load_train_val_test_csvs(dataset_name)
    print('Loaded dataset', dataset_name)

    #
    # specifying parameters
    n_components = 10
    # 'spherical', 'tied', 'diag', 'full'. Defaults to 'diag'.
    cov_type = 'diag'
    n_iters = 1000
    n_restarts = 10

    kwargs = {}
    kwargs['covariance_type'] = cov_type

    print('Clustering with GMM')
    clustering = algo.learnspn.cluster_rows(train,
                                            n_clusters=n_components,
                                            cluster_method='GMM',
                                            n_iters=n_iters,
                                            n_restarts=n_restarts,
                                            rand_gen=rand_gen,
                                            sklearn_args=kwargs)
    print('Clustering')
    print('numbers of clusters: ', len(clustering))

    assert len(clustering) == n_components

    tot_instances = 0
    for cluster in clustering:
        tot_instances += len(cluster)
        print('cluster length:', len(cluster))

    assert tot_instances == train.shape[0]


def test_cluster_rows_DPGMM():

    #
    # random generator
    seed = 1337
    rand_gen = numpy.random.RandomState(seed)

    #
    # loading a very simple dataset
    dataset_name = 'nltcs'
    train, valid, test = dataset.load_train_val_test_csvs(dataset_name)
    print('Loaded dataset', dataset_name)

    #
    # specifying parameters
    n_components = 1000
    # 'spherical', 'tied', 'diag', 'full'. Defaults to 'diag'.
    cov_type = 'diag'
    n_iters = 1000
    alpha = 4.0

    kwargs = {}
    kwargs['covariance_type'] = cov_type
    kwargs['verbose'] = True

    print('Clustering with DPGMM')
    clustering = algo.learnspn.cluster_rows(train,
                                            n_clusters=n_components,
                                            cluster_method='DPGMM',
                                            n_iters=n_iters,
                                            cluster_penalty=alpha,
                                            rand_gen=rand_gen,
                                            sklearn_args=kwargs)
    print('Clustering')
    print('numbers of clusters: ', len(clustering))
    tot_instances = 0
    for cluster in clustering:
        tot_instances += len(cluster)
        print('cluster length:', len(cluster))

    assert tot_instances == train.shape[0]


def test_learnspn_oneshot():

    logging.basicConfig(level=logging.INFO)
    #
    # loading a very simple dataset
    dataset_name = 'nltcs'
    train, valid, test = dataset.load_train_val_test_csvs(dataset_name)
    train_feature_vals = [2 for i in range(train.shape[1])]
    print('Loaded dataset', dataset_name)

    #
    # initing the algo
    learnSPN = algo.learnspn.LearnSPN(rand_gen=rand_gen)

    #
    # start learning
    spn = learnSPN.fit_structure(train,
                                 train_feature_vals)

    # print(spn)

    #
    # testing on-the-fly
    ll = 0.0
    for instance in test:
        ll += spn.single_eval(instance)[0]

    print('avg ll', ll / test.shape[0])


def test_learnspn_mixture_oneshot():

    logging.basicConfig(level=logging.INFO)
    #
    # loading a very simple dataset
    dataset_name = 'nltcs'
    train, valid, test = dataset.load_train_val_test_csvs(dataset_name)
    train_feature_vals = [2 for i in range(train.shape[1])]
    print('Loaded dataset', dataset_name)

    #
    # initing the algo
    learnSPN = algo.learnspn.LearnSPN(rand_gen=rand_gen)

    #
    # start learning
    n_mixtures = 10
    spns, (train_m_lls, valid_m_lls, test_m_lls) = \
        learnSPN.fit_mixture_bootstrap(train,
                                       n_mix_components=n_mixtures,
                                       valid=valid,
                                       test=test,
                                       feature_sizes=train_feature_vals)

    assert len(spns) == n_mixtures

    #
    # printing some stats
    print('TRAIN', train_m_lls.shape[0], train_m_lls.shape[1])
    print('VALID', valid_m_lls.shape[0], valid_m_lls.shape[1])
    print('TEST', test_m_lls.shape[0], test_m_lls.shape[1])

    assert train_m_lls.shape[0] == train.shape[0]
    assert valid_m_lls.shape[0] == valid.shape[0]
    assert test_m_lls.shape[0] == test.shape[0]

    train_m_file = 'train.m.lls.csv'
    valid_m_file = 'valid.m.lls.csv'
    test_m_file = 'test.m.lls.csv'

    #
    # reversing to csv
    numpy.savetxt(train_m_file, train_m_lls, delimiter=',', fmt='%.8e')
    numpy.savetxt(valid_m_file, valid_m_lls, delimiter=',', fmt='%.8e')
    numpy.savetxt(test_m_file, test_m_lls, delimiter=',', fmt='%.8e')


def test_learnspn_bagging_synth():
    n_features = 10
    n_instances = 50
    synth_data = numpy.random.binomial(1, 0.5, (n_instances, n_features))
    synth_features = numpy.array([2 for i in range(n_features)])

    logging.basicConfig(level=logging.DEBUG)

    #
    # initing the algo (default params)
    learnSPN = algo.learnspn.LearnSPN(rand_gen=rand_gen,
                                      min_instances_slice=2)

    #
    # fitting with bagging
    n_components = 4
    learn_start_t = perf_counter()
    spn = learnSPN.fit_structure_bagging(synth_data,
                                         synth_features,
                                         n_components)

    learn_end_t = perf_counter()
    print('Network learned in', learn_end_t - learn_start_t, 'secs')
    print(spn)


def test_spn_eval_opt():

    logging.basicConfig(level=logging.INFO)
    #
    # loading a very simple dataset
    dataset_name = 'nltcs'
    train, valid, test = dataset.load_train_val_test_csvs(dataset_name)
    train_feature_vals = [2 for i in range(train.shape[1])]
    print('Loaded dataset', dataset_name)

    #
    # initing the algo
    learnSPN = algo.learnspn.LearnSPN(rand_gen=rand_gen)

    learn_start_t = perf_counter()
    #
    # start learning
    spn = learnSPN.fit_structure(train,
                                 train_feature_vals)
    learn_end_t = perf_counter()
    print('Network learned in', (learn_end_t - learn_start_t), 'secs')

    # now checking performances
    infer_start_t = perf_counter()
    train_ll = 0.0
    print('Starting inference')
    for instance in train:
        (pred_ll, ) = spn.eval(instance)
        train_ll += pred_ll
    train_avg_ll = train_ll / train.shape[0]
    infer_end_t = perf_counter()
    # n avg ll -6.0180987340354 done in 43.947853350000514 secs
    print('train avg ll', train_avg_ll, 'done in',
          infer_end_t - infer_start_t, 'secs')


#
# to profile inference
if __name__ == '__main__':
    test_spn_eval_opt()
