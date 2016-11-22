from math import exp, log
#import numba

from spn.linked.nodes import ProductNode
from spn.linked.nodes import SumNode, PoissonNode


#@numba.jit
def eval_numba(nodes):
    for node in nodes:
        node.eval()


class Layer(object):

    """
    WRITEME
    """

    def __init__(self, nodes=None):
        """
        WRITEME
        """
        self._nodes = None
        self._n_nodes = None

        if nodes is None:
            self._nodes = []
            self._n_nodes = 0
        else:
            self._nodes = nodes
            self._n_nodes = len(nodes)

    def add_node(self, node):
        """
        WRITEME
        """
        self._nodes.append(node)
        self._n_nodes += 1

    def nodes(self):
        """
        WRITEME
        """
        for node in self._nodes:
            yield node

    def eval(self):
        """
        layer bottom-up evaluation
        """
        for node in self._nodes:
            node.eval()
        # eval_numba(self._nodes)

    def mpe_eval(self):
        """
        layer MPE bpttom-up evaluation
        """
        for node in self._nodes:
            node.mpe_eval()

    def backprop(self):
        """
        WRITEME
        """
        
        result = []
        for node in self._nodes:
            node.backprop()
            result.append(node.log_der)
        #print(result)
        #0/0
        return result

    def mpe_backprop(self):
        """
        WRITEME
        """
        for node in self._nodes:
            node.mpe_backprop()

    def set_log_derivative(self, log_der):
        """
        WRITEME
        """
        for node in self._nodes:
            node.log_der = log_der

    def node_values(self):
        """
        WRITEME
        """
        # depending on the freq of the op I could allocate
        # just once the list
        return [node.log_val for node in self._nodes]

    def get_nodes_by_id(self, node_pos):
        """
        this may be inefficient, atm used only in factory
        """
        node_list = [None for i in range(self._n_nodes)]
        for node in self._nodes:
            pos = node_pos[node.id]
            node_list[pos] = node
        return node_list

    def get_node(self, node_id):
        """
        WRITEME
        """
        return self._nodes[node_id]

    def n_nodes(self):
        """
        WRITEME
        """
        return self._n_nodes

    def n_edges(self):
        """
        WRITEME
        """
        edges = 0
        for node in self._nodes:
            # input layers have nodes with no children attr
            # try:
                # for child in node.children:
                #     edges += 1
            edges += node.n_children()
            # except:
            #     pass
        return edges

    def n_weights(self):
        """
        Only a sum layer has params
        """
        return 0

    def __repr__(self):
        """
        WRITEME
        """
        div = '\n**********************************************************\n'
        return '\n'.join([str(node) for node in self._nodes]) + div


class SumLayer(Layer):

    """
    WRITEME
    """

    def __init__(self, nodes=None):
        """
        WRITEME
        """
        Layer.__init__(self, nodes)

    def normalize(self):
        """
        WRITEME
        """
        for node in self._nodes:
            node.normalize()

    def add_edge(self, parent, child, weight):
        """
        WRITEME
        """
        parent.add_child(child, weight)

    # def update_weights(self, update_rule):
    #     """
    #     WRITEME
    #     """
    #     for node in self._nodes:
    #         weight_updates = [update_rule(weight,
    #                                       exp(child.log_val + node.log_der))
    #                           for child, weight
    #                           in zip(node.children, node.weights)]
    #         node.set_weights(weight_updates)

    def update_weights(self, update_rule, layer_id):
        """
        WRITEME
        """
        for node_id, node in enumerate(self._nodes):
            weight_updates = [update_rule(layer_id,
                                          node_id,
                                          weight_id,
                                          weight,
                                          exp(child.log_val + node.log_der))
                              for weight_id, (child, weight)
                              in enumerate(zip(node.children, node.weights))]
            node.set_weights(weight_updates)

    def is_complete(self):
        """
        WRITEME
        """
        return all([node.is_complete() for node in self.nodes()])

    def n_weights(self):
        """
        For a sum layer, its number of edges
        """
        return self.n_edges()

    def __repr__(self):
        return '[sum layer:]\n' + Layer.__repr__(self)


class ProductLayer(Layer):

    """
    WRITEME
    """

    def __init__(self, nodes=None):
        """
        WRITEME
        """
        Layer.__init__(self, nodes)

    def add_edge(self, parent, child):
        """
        WRITEME
        """
        parent.add_child(child)

    def is_decomposable(self):
        """
        WRITEME
        """
        return all([node.is_decomposable() for node in self.nodes()])

    def __repr__(self):
        return '[prod layer:]\n' + Layer.__repr__(self)


def compute_feature_vals(nodes):
    """
    From a set of input nodes, determine the feature ranges
    """
    feature_vals_dict = {}

    for node in nodes:
        if isinstance(node, PoissonNode):

            if node.var not in feature_vals_dict:
                feature_vals_dict[node.var] = node.var_val

    feature_vals = [feature_vals_dict[var]
                    for var in sorted(feature_vals_dict.keys())]

    return feature_vals

    

class PoissonLayer(Layer):

    """
    WRITEME
    """

    def __init__(self, nodes=None):
        """
        WRITEME
        """
        Layer.__init__(self, nodes)
        #self._vars = vars
        #self._feature_vals = compute_feature_vals(nodes)
        self._feature_vals = {}


    def eval(self, input=None):
        """
        WRITEME
        """
        
        if input is None:
            Layer.eval(self)
            return
        
        for node in self._nodes:
            # get the observed value
            obs = input[node.var]
            # and eval the node
            node.eval(obs)

    def vars(self):
        """
        WRITEME
        """
        return self._vars

    def feature_vals(self):
        """
        WRITEME
        """
        return self._feature_vals


    def add_edge(self, parent, child, weight):
        """
        WRITEME
        """
        parent.add_child(child, weight)

    # def update_weights(self, update_rule):
    #     """
    #     WRITEME
    #     """
    #     for node in self._nodes:
    #         weight_updates = [update_rule(weight,
    #                                       exp(child.log_val + node.log_der))
    #                           for child, weight
    #                           in zip(node.children, node.weights)]
    #         node.set_weights(weight_updates)

    def update_weights(self, update_rule, layer_id):
        """
        WRITEME
        """
        for node_id, node in enumerate(self._nodes):
            lambda_update = update_rule(layer_id,
                                          node_id,
                                          0,
                                          node.mean,
                                          #exp(node.log_der+log(max(float(node.obs)-float(node.mean), 0.0001))-log(float(node.mean))))
                                          exp(node.log_der) * ((float(node.obs)/float(node.mean)) - 1.0))
            if lambda_update <= 0:
                lambda_update = 0.0001
            
            node.mean = lambda_update
            
    def smooth_probs(self, alpha):
        pass

    def n_weights(self):
        """
        For a sum layer, its number of edges
        """
        return self.n_edges()

    def __repr__(self):
        return '[poisson layer:]\n' + Layer.__repr__(self)
    

class GaussianLayer(Layer):

    """
    WRITEME
    """

    def __init__(self, nodes=None):
        """
        WRITEME
        """
        Layer.__init__(self, nodes)
        #self._vars = vars
        #self._feature_vals = compute_feature_vals(nodes)
        self._feature_vals = {}


    def eval(self, input=None):
        """
        WRITEME
        """
        
        if input is None:
            Layer.eval(self)
            return
        
        for node in self._nodes:
            # get the observed value
            obs = input[node.var]
            # and eval the node
            node.eval(obs)

    def vars(self):
        """
        WRITEME
        """
        return self._vars

    def feature_vals(self):
        """
        WRITEME
        """
        return self._feature_vals


    def add_edge(self, parent, child, weight):
        """
        WRITEME
        """
        parent.add_child(child, weight)

    # def update_weights(self, update_rule):
    #     """
    #     WRITEME
    #     """
    #     for node in self._nodes:
    #         weight_updates = [update_rule(weight,
    #                                       exp(child.log_val + node.log_der))
    #                           for child, weight
    #                           in zip(node.children, node.weights)]
    #         node.set_weights(weight_updates)

    def update_weights(self, update_rule, layer_id):
        assert 0
            
    def smooth_probs(self, alpha):
        pass

    def n_weights(self):
        """
        For a sum layer, its number of edges
        """
        return self.n_edges()

    def __repr__(self):
        return '[gaussian layer:]\n' + Layer.__repr__(self)


class BernoulliLayer(Layer):

    """
    WRITEME
    """

    def __init__(self, nodes=None):
        """
        WRITEME
        """
        Layer.__init__(self, nodes)
        #self._vars = vars
        #self._feature_vals = compute_feature_vals(nodes)
        self._feature_vals = {}


    def eval(self, input=None):
        """
        WRITEME
        """

        if input is None:
            Layer.eval(self)
            return

        for node in self._nodes:
            # get the observed value
            obs = input[node.var]
            # and eval the node
            node.eval(obs)

    def vars(self):
        """
        WRITEME
        """
        return self._vars

    def feature_vals(self):
        """
        WRITEME
        """
        return self._feature_vals


    def add_edge(self, parent, child, weight):
        """
        WRITEME
        """
        parent.add_child(child, weight)

    # def update_weights(self, update_rule):
    #     """
    #     WRITEME
    #     """
    #     for node in self._nodes:
    #         weight_updates = [update_rule(weight,
    #                                       exp(child.log_val + node.log_der))
    #                           for child, weight
    #                           in zip(node.children, node.weights)]
    #         node.set_weights(weight_updates)

    def update_weights(self, update_rule, layer_id):
        assert 0

    def smooth_probs(self, alpha):
        pass

    def n_weights(self):
        """
        For a sum layer, its number of edges
        """
        return self.n_edges()

    def __repr__(self):
        return '[bernoulli layer:]\n' + Layer.__repr__(self)
