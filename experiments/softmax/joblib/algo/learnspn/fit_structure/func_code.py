# first line: 273
    def fit_structure(self, data):
        
        #
        # a queue containing the data slices to process
        slices_to_process = deque()

        # a stack for building nodes
        building_stack = deque()

        # a dict to keep track of id->nodes
        node_id_assoc = {}

        # check if there is only one family and map in to an array
        if isinstance(self.families, str):
            self.families = [self.families]*data.shape[1]
        # make sure families are numpy array
        self.families = numpy.asarray(self.families)
        self.config['families'] = self.families

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

                if self.families[current_features] == "poisson":
                    leaf_node = PoissonNode(data, current_instances, current_features)
                elif self.families[current_features] == "binomial":
                    leaf_node = BernoulliNode(data, current_instances, current_features)
                elif self.families[current_features] == "gaussian":
                    leaf_node = GaussianNode(data, current_instances, current_features)
                else:
                    assert False, "Nodetype not found"


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
                        #print(self.families[current_features])
                        #feature_clusters = retrieve_clustering(getIndependentGroupsStabilityTest(fcdata, families=self.families[current_features], alpha=self._alpha), indexes=current_features)
                        feature_clusters = retrieve_clustering(getIndependentGroups(fcdata, alpha=self._alpha, families=self.families[current_features]), indexes=current_features)



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
