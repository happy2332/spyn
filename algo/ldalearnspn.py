from collections import deque
import math
import numpy
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn.mixture
import sys

from algo.dataslice import DataSlice
from mlutils.LSH import above, make_planes
from mlutils.stabilityTest import getIndependentGroupsStabilityTest
from pdn.ABPDN import ABPDN

from spn import RND_SEED
from spn.factory import SpnFactory
from spn.linked.nodes import PoissonNode, GaussianNode
from spn.linked.nodes import ProductNode
from spn.linked.nodes import SumNode


try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time



NEG_INF = -sys.float_info.max





def retrieve_clustering(assignment, indexes=None):
    """
    from [2, 3, 8, 3, 1]
    to [{0}, {1, 3}, {2}, {3}]

    or

    from [2, 3, 8, 3, 1] and [21, 1, 4, 18, 11]
    to [{21}, {1, 18}, {4}, {11}]

    """

    clustering = []
    seen_clusters = dict()

    if indexes is None:
        indexes = [i for i in range(len(assignment))]

    for index, label in zip(indexes, assignment):
        if label not in seen_clusters:
            seen_clusters[label] = len(clustering)
            clustering.append([])
        clustering[seen_clusters[label]].append(index)

    return clustering


def cluster_rows(sliced_data,
                 n_clusters=2,
                 cluster_method='PDN',
                 n_iters=100,
                 n_restarts=3,
                 cluster_prep_method=None,
                 cluster_penalty=1.0,
                 rand_gen=None,
                 sklearn_args=None):
    """
    A wrapper to abstract from the implemented clustering method

    cluster_method = GMM | DPGMM | HOEM
    """

    clustering = None

    #
    # slicing the data


    # allIndexes = numpy.arange(0, sliced_data.shape[0])
    # zerorowsIdx = allIndexes[numpy.sum(sliced_data, 1) == 0]
    # datarowsIdx = allIndexes[numpy.sum(sliced_data, 1) > 0]
    # clustering_data = numpy.delete(sliced_data, zerorowsIdx, 0)
    clustering_data = sliced_data
    
    
    if cluster_prep_method == "tf-idf":
        tfidf_transformer = TfidfTransformer()
        clustering_data = tfidf_transformer.fit_transform(clustering_data)
    elif cluster_prep_method == "log+1":
        clustering_data = numpy.log(clustering_data + 1)
    elif cluster_prep_method == "sqrt":     
        clustering_data = numpy.sqrt(clustering_data)
        # clustering_data = clustering_data
        # row_sums = clustering_data.sum(axis=1) + 0.001
        # clustering_data = clustering_data / row_sums[:, numpy.newaxis]
        # clustering_data = numpy.sqrt(clustering_data)
    
    
    # sliced_data_sum = numpy.sum(sliced_data, axis=1)
    # sliced_data = sliced_data / sliced_data_sum[:, numpy.newaxis]    
    # sliced_data = numpy.sqrt(sliced_data)
    
    
    print("RUNNING CLUSTERING dims: " + str(sliced_data.shape) + " into: " + str(n_clusters) + " method: " + cluster_method + " pre: " + str(cluster_prep_method))
    #if sliced_data.shape[1] == 1:
    #    print("V" + str(data_slice.feature_ids))
        
    start_t = perf_counter()
    if cluster_method == 'PDN':
        
        assert cluster_prep_method == None
        
        clustering = ABPDN.pdnClustering(clustering_data, nM=n_clusters, maxIters=n_iters, max_depth=5)
        
            
    elif cluster_method == 'GMM':
        clustering_data = numpy.log(clustering_data + 1)
        #
        # retrieving other properties
        cov_type = sklearn_args['covariance_type'] \
            if 'covariance_type' in sklearn_args else 'diag'
        #
        # creating the cluster from sklearn
        gmm_c = sklearn.mixture.GMM(n_components=n_clusters,
                                    covariance_type=cov_type,
                                    random_state=rand_gen,
                                    n_iter=n_iters,
                                    n_init=n_restarts)

        #
        # fitting to training set
        try:
            gmm_c.fit(clustering_data)
        except Exception:
            pass

        #
        # getting the cluster assignment
        clustering = gmm_c.predict(clustering_data)
        
            
        
    elif cluster_method == "KMeans":
        clustering = KMeans(n_clusters=n_clusters, random_state=rand_gen, n_jobs=1).fit_predict(clustering_data)
            
    elif cluster_method == "RandomPartition":
        clustering = above(make_planes(1, clustering_data.shape[1]), clustering_data)[:, 0]

    elif cluster_method == 'DPGMM':
        #
        # retrieving other properties
        cov_type = sklearn_args['covariance_type'] \
            if 'covariance_type' in sklearn_args else 'diag'
        verbose = sklearn_args['verbose']\
            if 'verbose' in sklearn_args else False

        dpgmm_c = sklearn.mixture.DPGMM(n_components=n_clusters,
                                        covariance_type=cov_type,
                                        random_state=rand_gen,
                                        n_iter=n_iters,
                                        alpha=cluster_penalty,
                                        verbose=verbose)

        #
        # fitting to training set
        dpgmm_c.fit(clustering_data)

        #
        # getting the cluster assignment
        clustering = dpgmm_c.predict(clustering_data)

    elif cluster_method == 'HOEM':
        raise NotImplementedError('Hard Online EM is not implemented yet')
    else:
        raise Exception('Clustering method not valid')
    
    end_t = perf_counter()
    
    print('Clustering done in %f secs' % (end_t - start_t))
    
#    nI = sliced_data.shape[0]
#    uniqueNi = len(set([tuple(x) for x in clustering_data]))
#    print(nI, uniqueNi, sum(clustering))
    # guarantee that we have a partition
    #if sum(clustering) == 0:
        #split evenly in n clusters
    #    clustering = numpy.asarray((list(range(n_clusters))*math.ceil(nI/n_clusters))[0:nI])
    
    # print(sliced_data)
    print(list(map(lambda c: numpy.sum(clustering == c), range(n_clusters))))

    # clusteringComplete = numpy.zeros(data_slice.instance_ids.shape)
    # clusteringComplete[zerorowsIdx] = n_clusters
    # clusteringComplete[datarowsIdx] = clustering
    # return retrieve_clustering(clustering, data_slice.instance_ids[datarowsIdx])
    return clustering
    #return retrieve_clustering(clustering, data_slice.instance_ids)



class LearnSPN(object):

    """
    Implementing Gens and Domingos
    """

    def __init__(self,
                 min_instances_slice=50,
                 min_features_slice=0,
                 alpha=0.001,
                 row_cluster_method='KMeans',
                 ind_test_method="pairwise_treeglm",
                 sub_sample_rows=1000,
                 cluster_penalty=2.0,
                 n_cluster_splits=2,
                 n_iters=1000,
                 n_restarts=2,
                 sklearn_args={},
                 cltree_leaves=False,
                 poisson_leaves=True,
                 rand_gen=None,
                 cluster_prep_method="sqrt",
                 family="poisson", 
                 cluster_first=True,
                 cache=None):
        """
        WRITEME
        """
        self._min_instances_slice = min_instances_slice
        self._min_features_slice = min_features_slice
        self._alpha = alpha
        self._row_cluster_method = row_cluster_method
        self._ind_test_method = ind_test_method
        self._cluster_penalty = cluster_penalty
        self._n_cluster_splits = n_cluster_splits
        self._n_iters = n_iters
        self._n_restarts = n_restarts
        self._sklearn_args = sklearn_args
        self._cltree_leaves = cltree_leaves
        self.poisson_leaves = poisson_leaves
        self._cluster_prep_method = cluster_prep_method
        self.family = family
        self._sub_sample_rows = sub_sample_rows
                
        self._cluster_first = cluster_first
        
        self._rand_gen = rand_gen if rand_gen is not None \
            else numpy.random.RandomState(RND_SEED)
            
        if cache is not None:
            self.fit_structure = cache.cache(self.fit_structure)
            
            
            
        self.config = {"min_instances":min_instances_slice,
                       "alpha":alpha,
                       "cluster_method":row_cluster_method,
                       "cluster_n_clusters":n_cluster_splits,
                       "cluster_iters": n_iters,
                       "cluster_prep_method": cluster_prep_method,
                       "family": self.family
                       }


        #
        # resetting the data slice ids (just in case)
        DataSlice.reset_id_counter()

    

    def fit_structure(self, data):
        
        #
        # a queue containing the data slices to process
        slices_to_process = deque()

        # a stack for building nodes
        building_stack = deque()

        # a dict to keep track of id->nodes
        node_id_assoc = {}

        # creating the first slice
        whole_slice = DataSlice.whole_slice(data.shape[0], data.shape[1])
        slices_to_process.append(whole_slice)

        cluster_first = self._cluster_first
        
        #
        # iteratively process & split slices
        #
        while slices_to_process:

            # process a slice
            current_slice = slices_to_process.popleft()

            # pointers to the current data slice
            current_instances = current_slice.instance_ids
            current_features = current_slice.feature_ids
            current_id = current_slice.id

            n_features = len(current_features)
            
#             if n_features > 1:
# #                 # print("removing Zeros")
#                 datarowsIdx = numpy.sum(data[current_instances, :][:, current_features], 1) > 0
#                 if not any(datarowsIdx):
#                     datarowsIdx[0] = True
#                 current_instances = current_slice.instance_ids[datarowsIdx]
            
            n_instances = len(current_instances)

#             if n_instances == 0:
#                 #too strong cutting the zeroes
#                 current_instances = [current_slice.instance_ids[0]]
#                 n_instances = len(current_instances)

            slice_data_rows = data[current_instances, :]
            current_slice_data = slice_data_rows[:, current_features]
            
            # is this a leaf node or we can split?
            if n_features == 1 and (current_slice.doNotCluster or n_instances <= self._min_instances_slice):

                (feature_id,) = current_features
                
                if self.family == "poisson":
                    leaf_node = PoissonNode(data, current_instances, current_features)
                elif self.family == "gaussian":
                    leaf_node = GaussianNode(data, current_instances, current_features)
                
                # storing links
                # input_nodes.append(leaf_node)
                leaf_node.id = current_id
                node_id_assoc[current_id] = leaf_node


            # elif (current_slice_data.shape[0] < self._min_instances_slice):
            # elif ( (n_instances <= self._min_instances_slice and n_features > 1) and current_slice_data.shape[0]  < self._min_instances_slice):
            # elif ((n_instances <= self._min_instances_slice and n_features > 1)):
            elif n_features > 1 and (current_slice.doNotCluster or n_instances <= self._min_instances_slice)  :
            
                
                # print('into naive factorization')
                child_slices = [DataSlice(current_instances, [feature_id]) for feature_id in current_features]
                slices_to_process.extend(child_slices)
 
                #children_ids = [child.id for child in child_slices]
 
                for child_slice in child_slices:
                    child_slice.doNotCluster = current_slice.doNotCluster
                    current_slice.add_child(child_slice)
                current_slice.type = ProductNode
                building_stack.append(current_slice)
 
                prod_node = ProductNode(data, current_instances, current_features)
                prod_node.id = current_id
 
                node_id_assoc[current_id] = prod_node
 
            else:

                split_on_features = False
                
                # first_run = False
                #
                # first run is a split on rows
                if n_features == 1 or cluster_first :
                    cluster_first = False
                else:

                    if self._ind_test_method=="pairwise_treeglm"or self._ind_test_method=="subsample":
                         
                        fcdata = current_slice_data
                        
                        if self._ind_test_method=="subsample":
                            #sampled_rows = 2000
                            #sampled_rows = math.floor(current_slice_data.shape[0]*10/100)
                            sampled_rows = self._sub_sample_rows
                            if sampled_rows < current_slice_data.shape[0]:
                                fcdata = current_slice_data[numpy.random.choice(current_slice_data.shape[0], sampled_rows, replace=False)]
                            else:
                                fcdata = current_slice_data
                        
                        
                        #Using R
                        #from pdn.independenceptest import getIndependentGroups
                        #feature_clusters = retrieve_clustering(getIndependentGroups(fcdata, alpha=self._alpha, family=self.family), current_features)
                        feature_clusters = retrieve_clustering(getIndependentGroupsStabilityTest(fcdata, alpha=self._alpha), current_features)
                    elif self._ind_test_method=="KMeans" :
                        
                        feature_clusters = retrieve_clustering(cluster_rows((data[current_instances, :][:, current_features]).T,
                                     n_clusters=2,
                                     cluster_method=self._row_cluster_method,
                                     n_iters=self._n_iters,
                                     n_restarts=self._n_restarts,
                                     cluster_prep_method="sqrt",
                                     cluster_penalty=self._cluster_penalty,
                                     rand_gen=self._rand_gen,
                                     sklearn_args=self._sklearn_args), current_instances)
                    
                    split_on_features = len(feature_clusters) > 1
                    
                #
                # have dependent components been found?
                if split_on_features:
                    #
                    # splitting on columns
                    # print('---> Splitting on features')
                    # print(feature_clusters)


                    slices = [DataSlice(current_instances, cluster) for cluster in feature_clusters]
    
                    slices_to_process.extend(slices)
    
                    current_slice.type = ProductNode
                    building_stack.append(current_slice)
                    for child_slice in slices:
                        current_slice.add_child(child_slice)
    
                    prod_node = ProductNode(data, current_instances, current_features)
                    prod_node.id = current_id
                    node_id_assoc[current_id] = prod_node
                    


                else:
                    # print('---> Splitting on rows')

                    k_row_clusters = min(self._n_cluster_splits, n_instances - 1)

                    if n_features == 1:
                        # do one kmeans run with K large enough to split into N min instances
                        k_row_clusters = math.floor(n_instances / self._min_instances_slice) + 1
                        k_row_clusters = min(k_row_clusters, n_instances - 1)

                    clustering = retrieve_clustering(cluster_rows(data[current_instances, :][:, current_features],
                                     n_clusters=k_row_clusters,
                                     cluster_method=self._row_cluster_method,
                                     n_iters=self._n_iters,
                                     n_restarts=self._n_restarts,
                                     cluster_prep_method=self._cluster_prep_method,
                                     cluster_penalty=self._cluster_penalty,
                                     rand_gen=self._rand_gen,
                                     sklearn_args=self._sklearn_args), current_instances)
                    
                    
                    cluster_slices = [DataSlice(cluster, current_features) for cluster in clustering]
                    
                    if len(clustering) < k_row_clusters:
                        for cluster_slice in cluster_slices:
                            cluster_slice.doNotCluster = True
                    
                    
                    n_instances_clusters = sum([len(cluster) for cluster in clustering])
                    cluster_weights = [len(cluster) / n_instances_clusters for cluster in clustering]
    
                    slices_to_process.extend(cluster_slices)
    
                    current_slice.type = SumNode
                    building_stack.append(current_slice)
                    for child_slice, child_weight in zip(cluster_slices, cluster_weights):
                        current_slice.add_child(child_slice, child_weight)
    
                    sum_node = SumNode(data, current_instances, current_features)
                    sum_node.id = current_id
                    node_id_assoc[current_id] = sum_node



        root_node = SpnFactory.pruned_spn_from_slices(node_id_assoc, building_stack, True)
        
        spn = SpnFactory.layered_linked_spn(root_node, data, self.config)


        return spn

    

    
