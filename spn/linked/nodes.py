from math import exp, floor
from math import log
from statistics import mean
import sys

import numpy

from mlutils.statistics import logpoissonpmf, loggaussianpdf, poissonpmf
from mlutils.statistics import gaussianpdf, loggaussianpdf
from mlutils.statistics import bernoullipmf, logbernoullipmf

from spn import IS_LOG_ZERO
from spn import LOG_ZERO
from spn import MARG_IND
from spn import utils


# import numba
NODE_SYM = 'u'  # unknown type
SUM_NODE_SYM = '+'
PROD_NODE_SYM = '*'
INDICATOR_NODE_SYM = 'i'
DISCRETE_VAR_NODE_SYM = 'd'
CHOW_LIU_TREE_NODE_SYM = 'c'
POISSON_VAR_NODE_SYM = 'p'
GAUSSIAN_VAR_NODE_SYM = 'g'
BERNOULLI_VAR_NODE_SYM = 'b'

class Node(object):


    # class id counter
    id_counter = 0

    def __init__(self, data, instances, features):

        # default val is 0.
        self.log_val = LOG_ZERO
        
        self.log_mpeval = -sys.float_info.max

        # setting id and incrementing
        self.id = Node.id_counter
        Node.id_counter += 1

        # derivative computation
        self.log_der = LOG_ZERO

        self.data = data
        self.instances = numpy.copy(instances)
        self.instances.flags.writeable = False
        
        self.features = numpy.copy(features)
        self.features.flags.writeable = False
        
        self.children = []
        

    def __repr__(self):
        return 'id: {id} scope: {features}'.format(id=self.id,
                                                features=self.features)

    # this is probably useless, using it for test purposes
    def set_val(self, val):

        if numpy.allclose(val, 0, 1e-10):
            self.log_val = LOG_ZERO
        else:
            self.log_val = log(val)

    def __hash__(self):
        """
        A node has a unique id
        """
        return hash(self.id)

    def __eq__(self, other):

        return self.id == other.id


    def depth(self):
        return 1

    def node_type_str(self):
        return NODE_SYM

    #def node_graph_str(self, fnames, astopic=False):
    #    return self.node_type_str()

    def node_graph_str(self, fnames, astopic=False):
        if not astopic:
            return self.node_type_str()
        
        # compute multinomial
        #features = [n.features[0] for n in self.children if isinstance(n, PoissonNode)]
        #features = [n.features[0] for n in self.children]
        features = self.features
        
        current_data = self.data[self.instances, :][:, features]
        
        theta = numpy.mean(current_data, 0)
        m = numpy.sum(theta)
        if m != 0:
            print(m, theta)
            theta = theta / m
        
        thetaindexes = numpy.argsort(theta)[::-1][0:10]
        
        fnames = numpy.asarray(fnames)[features][thetaindexes]
        theta = theta[thetaindexes]
        
        
        
        return ("%s\n%s\n(i=%s,f=%s/%s)\n" % (self.node_type_str(), round(m, 4), current_data.shape[0], current_data.shape[1], len(self.features))) + "\n".join(["%s=%s" % (feature, round(th, 4)) for feature, th in zip(fnames, theta)])


    def node_short_str(self):
        return "{0} {1}\n".format(self.node_type_str(),
                                  self.id)


    def n_children(self):
        return len(self.children)

    def is_topic_node(self):
        return False 

    @classmethod
    def reset_id_counter(cls):

        Node.id_counter = 0


    def size(self):
        return 1 + sum([c.size() for c in self.children])


# @numba.njit

def eval_sum_node(children_log_vals, log_weights):
    """
    numba version
    """

    max_log = LOG_ZERO

    n_children = children_log_vals.shape[0]

    # getting the max
    for i in range(n_children):
        ch_log_val = children_log_vals[i]
        log_weight = log_weights[i]
        w_sum = ch_log_val + log_weight
        if w_sum > max_log:
            max_log = w_sum

    # log_unnorm = LOG_ZERO
    # max_child_log = LOG_ZERO

    sum_val = 0.
    for i in range(n_children):
        ch_log_val = children_log_vals[i]
        log_weight = log_weights[i]
        # for node, log_weight in zip(children, log_weights):
        # if node.log_val is False:
        ww_sum = ch_log_val + log_weight
        sum_val += exp(ww_sum - max_log)

    # is this bad code?
    log_val = LOG_ZERO
    if sum_val > 0.:
        log_val = log(sum_val) + max_log

    return log_val
    # log_unnorm = log(sum_val) + max_log
    # self.log_val = log_unnorm - numpy.log(self.weights_sum)
    # return self.log_val


class SumNode(Node):

    def __init__(self, data, instances, features):

        Node.__init__(self, data, instances, features)
        self.weights = []
        self.log_weights = []
        self.weights_sum = 0

    def add_child(self, child, weight):

        self.children.append(child)
        self.weights.append(weight)
        self.log_weights.append(log(weight))
        self.weights_sum += weight

    def set_weights(self, weights):


        self.weights = weights

        # normalizing self.weights
        w_sum = sum(self.weights)
        for i, weight in enumerate(self.weights):
            self.weights[i] = weight / w_sum

        # updating log weights
        for i, weight in enumerate(weights):
            self.log_weights[i] = log(weight) if weight > 0.0 else LOG_ZERO

        # and also the sum
        self.weights_sum = sum(weights)

    # @numba.jit
    def eval(self):

        # resetting the log derivative
        self.log_der = LOG_ZERO

        max_log = LOG_ZERO

        # getting the max
        for node, log_weight in zip(self.children, self.log_weights):
            w_sum = node.log_val + log_weight
            if w_sum > max_log:
                max_log = w_sum

        # log_unnorm = LOG_ZERO
        # max_child_log = LOG_ZERO

        sum_val = 0.
        for node, log_weight in zip(self.children, self.log_weights):
            # if node.log_val is False:
            ww_sum = node.log_val + log_weight
            sum_val += exp(ww_sum - max_log)

        # is this bad code?
        if sum_val > 0.:
            self.log_val = log(sum_val) + max_log

        else:
            self.log_val = LOG_ZERO

        # # up to now numba

        # log_unnorm = log(sum_val) + max_log
        # self.log_val = log_unnorm - numpy.log(self.weights_sum)
        # return self.log_val

        # self.log_val = eval_sum_node(numpy.array([child.log_val
        #                                           for child in self.children]),
        #                              numpy.array(self.log_weights))

    def mpe_eval(self):

        # resetting the log derivative
        self.log_der = LOG_ZERO

        # log_val is used as an accumulator, one less var
        self.log_val = LOG_ZERO

        # getting the max
        for node, log_weight in zip(self.children, self.log_weights):
            w_sum = node.log_val + log_weight
            if w_sum > self.log_val:
                self.log_val = w_sum


    def complete(self, data):
        self.log_mpeval = -sys.float_info.max
        
        result = None
        for node in self.children:
            # (data2, path2) = node.complete(data)
            data2 = node.complete(data)
            
            if node.log_mpeval > self.log_mpeval:
                self.log_mpeval = node.log_mpeval
                # result = (data2, path2)
                result = data2
            
        return result

    def marginalizeToEquation(self, features, evidence, fmt="python", ctx=None):


        #weights = numpy.round(self.weights, 4)
        weights = self.weights
        
        weights[-1] = 1.0 - sum(weights[0:-1])
        
        if fmt == "cuda1":
            result = ""
            for i, c in enumerate(self.children):
                ceq = c.marginalizeToEquation(features, evidence, fmt, ctx)
                if len(ceq) > ctx["max"]:
                    ctx["pre"].append("tmpstack[%s] = %s" %(ctx["id"], ceq))
                    result += "+(%s*tmpstack[%s])" %(weights[i], ctx["id"])
                    ctx["id"] += 1
                else: 
                    result += "+(%s*%s)" %(weights[i], ceq)
                
                if len(result) > ctx["max"]:
                    ctx["pre"].append("tmpstack[%s] = %s" %(ctx["id"], result[1:]))
                    result = "+tmpstack[%s]"%(ctx["id"])
                    ctx["id"] += 1
            return result[1:]

        return "(" + " + ".join(map(lambda i: str(weights[i]) + "*(" + self.children[i].marginalizeToEquation(features, evidence, fmt) + ")", range(len(self.children)))) + ")"

       
        

    def backprop(self):
        """
        WRITE
        """
        # if it is not zero we can pass
        if self.log_der > LOG_ZERO:
            # dS/dS_n = sum_{p}: dS/dS_p * dS_p/dS_n
            # per un nodo somma p
            #
            for child, log_weight in zip(self.children, self.log_weights):
                # print('child before', child.log_der)
                # if child.log_der == LOG_ZERO:
                # if IS_LOG_ZERO(child.log_der):
                if child.log_der <= LOG_ZERO:
                    child.log_der = self.log_der + log_weight
                else:
                    child.log_der = numpy.logaddexp(child.log_der,
                                                    self.log_der + log_weight)
                # print('child after', child.log_der)
        # update weight log der too ?

    def mpe_backprop(self):

        if self.log_der > LOG_ZERO:
            # the child der is the max der among parents
            for child in self.children:
                child.log_der = max(child.log_der, self.log_der)

    def normalize(self):

        # normalizing self.weights
        w_sum = sum(self.weights)
        for i, weight in enumerate(self.weights):
            self.weights[i] = weight / w_sum

        # computing log(self.weights)
        for i, weight in enumerate(self.weights):
            self.log_weights[i] = log(weight) if weight > 0.0 else LOG_ZERO

    def is_complete(self):

        _complete = True
        # all children scopes shall be equal
        children_scopes = [child.features
                           for child in self.children]

        # adding this node scope
        children_scopes.append(self.features)

        for scope1, scope2 in utils.pairwise(children_scopes):
            if scope1 != scope2:
                _complete = False
                break

        return _complete



    def depth(self):
        return 1 + max(map(lambda c: c.depth(), self.children))

    def node_type_str(self):
        return "XOR"

    def node_short_str(self):
        children_str = " ".join(["{id}:{weight:.8f}".format(id=node.id,
                                                        weight=weight)
                                 for node, weight in zip(self.children,
                                                         self.weights)])
        return "{type} {id} [{children}]".format(type=self.node_type_str(),
                                                 id=self.id,
                                                 children=children_str)

    def __repr__(self):
        base = Node.__repr__(self)
        children_info = [(node.id, weight)
                         for node, weight in zip(self.children,
                                                 self.weights)]
        msg = ''
        for id, weight in children_info:
            msg += ' ({id} {weight:.8f})'.format(id=id,
                                             weight=weight)
        return 'Sum Node {line1}\n{line2}'.format(line1=base,
                                                  line2=msg)


# @numba.njit

def eval_prod_node(children_log_vals):


    n_children = children_log_vals.shape[0]

    # and the zero children counter
    # zero_children = 0

    # computing the log value
    log_val = 0.0
    for i in range(n_children):
        ch_log_val = children_log_vals[i]
        # if ch_log_val <= LOG_ZERO:
        #     zero_children += 1

        log_val += ch_log_val

    return log_val  # , zero_children


class ProductNode(Node):



    def __init__(self, data, instances, features):

        Node.__init__(self, data, instances, features)
        # bit for zero children, see Darwiche
        self.zero_children = 0

    def add_child(self, child):

        self.children.append(child)

    def eval(self):

        # resetting the log derivative
        self.log_der = LOG_ZERO

        # and the zero children counter
        self.zero_children = 0

        # computing the log value
        self.log_val = 0.0
        for node in self.children:
            if node.log_val <= LOG_ZERO:
                self.zero_children += 1

            self.log_val += node.log_val

        #
        # numba
        # self.log_val = \
        #     eval_prod_node(numpy.array([child.log_val
        #                                 for child in self.children]))
        # return self.log_val

    def mpe_eval(self):
        """
        Just redirecting normal evaluation
        """
        self.eval()


    def marginalizeToEquation(self, features, evidence=None, fmt="python", ctx=None):

        if fmt == "cuda1":
            result = ""
            for i, c in enumerate(self.children):
                ceq = c.marginalizeToEquation(features, evidence, fmt, ctx)
                if len(ceq) > ctx["max"]:
                    ctx["pre"].append("tmpstack[%s] = %s" %(ctx["id"], ceq))
                    result += "*tmpstack[%s]" %(ctx["id"])
                    ctx["id"] += 1
                else: 
                    result += "*%s" %(ceq)
                    
                if len(result) > ctx["max"]:
                    ctx["pre"].append("tmpstack[%s] = %s" %(ctx["id"], result[1:]))
                    result = "*tmpstack[%s]"%(ctx["id"])
                    ctx["id"] += 1
            return result[1:]

        return "(" + " * ".join(map(lambda child:  child.marginalizeToEquation(features, evidence, fmt), self.children)) + ")"



    def backprop(self):

        if self.log_der > LOG_ZERO:

            for child in self.children:
                log_der = LOG_ZERO
                # checking the bit
                if self.zero_children == 0:
                    log_der = self.log_val - child.log_val
                elif self.zero_children == 1 and child.log_val <= LOG_ZERO:
                    log_der = sum([node.log_val for node in self.children
                                   if node != child])
                    # log_der = 0.0
                    # for node in self.children:
                    #     if node != child:
                    #         log_der += node.log_val
                # adding this parent value
                log_der += self.log_der
                # if child.log_der <= LOG_ZERO:
                # if IS_LOG_ZERO(child.log_der):
                if child.log_der <= LOG_ZERO:
                    # first assignment
                    child.log_der = log_der
                else:
                    child.log_der = numpy.logaddexp(child.log_der,
                                                    log_der)

    def mpe_backprop(self):

        if self.log_der > LOG_ZERO:
            for child in self.children:
                log_der = LOG_ZERO
                # checking the bit
                if self.zero_children == 0:
                    log_der = self.log_val - child.log_val
                elif self.zero_children == 1 and child.log_val <= LOG_ZERO:
                    log_der = sum([node.log_val for node in self.children
                                   if node != child])
                # adding this parent value
                log_der += self.log_der
                # updating child log der with the max instead of sum
                child.log_der = max(child.log_der, log_der)


    def complete(self, data):
        data = numpy.copy(data)
        # path = {}
        
        self.log_mpeval = 0
        
        for node in self.children:
            # (data2, path2) = node.complete(data)
            data2 = node.complete(data)
            self.log_mpeval += node.log_mpeval
            
            idx = data != data2
            data[idx] = data2[idx]
            # for i in numpy.where(idx)[0]:
            #    if i not in path:
            #        path[i] = []
            #    path[i].extend(path2[i])
            #    path[i].insert(0,self.id)
        
        
        if self.log_mpeval == 0:
            self.log_mpeval = -sys.float_info.max
        
        # return (data, path)
        return data


    def backprop2(self):

        # if more than one child has a zero value, cannot propagate
        if self.log_val <= LOG_ZERO:
            count = 0
            for child in self.children:
                if child.log_val <= LOG_ZERO:
                    count += 1
                    if count > 1:
                        return

        # only when needed
        if self.log_der > LOG_ZERO:
            for child in self.children:
                # print('b child val', child.log_val, child.log_der)
                if child.log_val <= LOG_ZERO:
                    # print('child log zero')
                    # shall loop on other children
                    # maybe this is memory consuming, but shall be faster
                    # going to numpy array shall be faster
                    log_der = sum([node.log_val for node in self.children
                                   if node.log_val > LOG_ZERO]) + \
                        self.log_der
                    if child.log_der <= LOG_ZERO:
                        # print('first log, add', log_der)
                        child.log_der = log_der
                    else:

                        child.log_der = numpy.logaddexp(child.log_der,
                                                        log_der)
                        # print('not first log, added', child.log_der)
                # if it is 0 there is no point updating children
                elif self.log_val > LOG_ZERO:
                    # print('par val not zero')
                    if child.log_der <= LOG_ZERO:
                        child.log_der = self.log_der + \
                            self.log_val - \
                            child.log_val
                        # print('child val not zero', child.log_der)
                    else:
                        child.log_der = numpy.logaddexp(child.log_der,
                                                        self.log_der + 
                                                        self.log_val - 
                                                        child.log_val)
                        # print('child log der not first', child.log_der)

    def is_decomposable(self):

        decomposable = True
        whole = set()
        for child in self.children:
            child_scope = child.features
            for scope_var in child_scope:
                if scope_var in whole:
                    decomposable = False
                    break
                else:
                    whole.add(scope_var)
            else:
                continue
            break

        if whole != self.features:
            decomposable = False
        return decomposable

    def n_children(self):
        return len(self.children)

    def is_topic_node(self):
        return all([isinstance(n, PoissonNode) for n in self.children])

    def depth(self):
        return 1 + max(map(lambda c: c.depth(), self.children))

    def node_type_str(self):
        return PROD_NODE_SYM

    def node_short_str(self):
        children_str = " ".join(["{id}".format(id=node.id)
                                 for node in self.children])
        return "{type} {id} [{children}]".format(type=self.node_type_str(),
                                                 id=self.id,
                                                 children=children_str)


    def node_graph_str(self, fnames, astopic=False):
        if not astopic:
            return self.node_type_str()
        
        # compute multinomial
        #features = [n.features[0] for n in self.children if isinstance(n, PoissonNode)]
        features = [n.features[0] for n in self.children]
        
        current_data = self.data[self.instances, :][:, features]
        
        theta = numpy.mean(current_data, 0)
        m = numpy.sum(theta)
        if m != 0:
            print(m, theta)
            theta = theta / m
        
        thetaindexes = numpy.argsort(theta)[::-1][0:10]
        
        fnames = numpy.asarray(fnames)[features][thetaindexes]
        theta = theta[thetaindexes]
        
        
        
        return ("%s\n(i=%s,f=%s/%s)\n" % (round(m, 4), current_data.shape[0], current_data.shape[1], len(self.features))) + "\n".join(["%s=%s" % (feature, round(th, 4)) for feature, th in zip(fnames, theta)])

    def __repr__(self):
        base = Node.__repr__(self)
        children_info = [node.id
                         for node in self.children]
        msg = ''
        for id in children_info:
            msg += ' ({id})'.format(id=id)
        return 'Prod Node {line1}\n{line2}'.format(line1=base,
                                                   line2=msg)




class PoissonNode(Node):

    def __init__(self, data, instances, features):

        Node.__init__(self, data, instances, features)

        self.var = list(self.features)[0]
        
        self.instances = data.shape[0]
        self.mean = numpy.mean(data[instances, self.var])    

        if self.mean == 0:
            self.mean = 0.01
        
        self._instances = instances
        self.data = data

    def backprop(self):
        # because the poisson node is both and input node and a node on the bottom layer, we don't need to back propagate to children nodes, there are none
        pass


    def marginalizeToEquation(self, features, evidence, fmt="python", ctx=None):
        
        if self.var not in features:
            if evidence is None or numpy.isnan(evidence[self.var]):
                return "1.0"
            
            return "%.30f" % (poissonpmf(evidence[self.var], self.mean))
        
        # return "((e^(-%s)*%s^(x_%s))/x_%s!)"%(round(self.mean,4), round(self.mean,4), self.var, self.var)
        
        #return "((exp(-%s)*%s**(x_%s))/factorial(x_%s))" % (round(self.mean, 4), round(self.mean, 4), self.var, self.var)
        #return "((math.exp(-{mean})*{mean}**(x_{var}_))/math.factorial(x_{var}_))".format(**{'mean': self.mean, 'var': self.var})
        
        #return "(({expminusmean}*({mean}**(x_{var}_)))/math.factorial(x_{var}_))".format(**{'expminusmean': exp(-self.mean), 'mean': self.mean, 'var': self.var})
        
        if fmt == "python":
            return "poissonpmf(x_{var}_, {mean})".format(**{'mean': self.mean, 'var': self.var})
        elif fmt == "numpy":
            return "nppoissonpmf(x[:, {var}], {mean})".format(**{'mean': self.mean, 'var': self.var})
        elif fmt == "mathematica":
            return "PDF[PoissonDistribution[{mean}], x_{var}_]".format(**{'mean': self.mean, 'var': self.var})
        elif fmt == "cuda":
            #return "math.exp(-math.lgamma(inp[i,{var}] + 1.0) - {mean} + inp[i,{var}] * math.log({mean}))".format(**{'mean': self.mean, 'var': self.var})
            return "cuda_poissonpmf(inp[{var}], {mean})".format(**{'mean': self.mean, 'var': self.var})
        else: 
            assert False, "marginalizeToEquation"
        

    def smooth_probs(self, alpha, data=None):
        pass

    def eval(self, obs=None):
        if obs is None:
            return

        self.log_der = LOG_ZERO
        self.obs = obs
        
        self.log_val = logpoissonpmf(obs, self.mean)
        #print("POIS", self.log_val, obs, self.mean)

    def mpe_eval(self, obs):
        return 1.0
  
    def complete(self, data):
        data = numpy.copy(data)
        
        # path = {}
        
        obs = data[self.var]
        if numpy.isnan(obs):
            obs = floor(self.mean)
            data[self.var] = obs
            # path[self.var] = [self.id]
        
        #print("POIS", obs, self.mean)
        self.log_mpeval = logpoissonpmf(obs, self.mean)
        
        # return (data, path)
        return data
        
        
  
    def n_children(self):
        return 0


    
    def node_type_str(self):
        return POISSON_VAR_NODE_SYM

    def node_graph_str(self, fnames, astopic=False):
        return "%s%s~Pois_%s(%s)" % (fnames[self.var], self.var, self.instances, round(self.mean, 3))

    def node_short_str(self, features=None):
        mean_str = str(self.mean)
        
        vars = self.var
        
        if features is not None:
            vars = features[vars]
        
        return "{type} {id} <{vars}> {mean}".format(type=self.node_type_str(), id=self.id, vars=vars, mean=mean_str)

    def __repr__(self):
        base = Node.__repr__(self)

        return ("""Poisson Node {line1}
            var: {var} data: [{data}] mean: [{mean}] real_mean: [{rmean}]""".
                format(line1=base,
                       var=self.var,
                       data=self.data.ravel(),
                       mean=self.mean,
                       rmean=numpy.mean(self.data.ravel())))

    def var_values(self):
        return len(self._var_freqs)
    
class GaussianNode(Node):

    def __init__(self, data, instances, features):

        Node.__init__(self, data, instances, features)

        self.var = list(self.features)[0]
        
        self.instances = data.shape[0]
        self.mean = numpy.mean(data[instances, self.var])
        self.variance = numpy.var(data[instances, self.var])
        # print(self.var, data, self.instances, data[instances, self.var])
        assert self.variance >= 0, self.variance
        
        if self.variance == 0:
            #self.variance = 0.1
            self.variance = 0.0001
        
        self._instances = instances
        self.data = data

    def backprop(self):
        # because the gaussian node is both and input node and a node on the bottom layer, we don't need to back propagate to children nodes, there are none
        pass


    def marginalizeToEquation(self, features, evidence, fmt="python", ctx=None):

        if self.var not in features:
            if evidence is None or numpy.isnan(evidence[self.var]):
                return "1.0"

            return "%.30f" % (gaussianpdf(evidence[self.var], self.mean, self.variance))
        
        # return "((e^(-%s)*%s^(x_%s))/x_%s!)"%(round(self.mean,4), round(self.mean,4), self.var, self.var)
        
        # return "((exp(-%s)*%s**(x_%s))/factorial(x_%s))" % (round(self.mean, 4), round(self.mean, 4), self.var, self.var)

        if fmt == "python":
            # TODO is still numpy
            return "gaussianpdf(x_{var}_, {mean}, {variance})".format(**{'mean': self.mean, 'var': self.var, 'variance': self.variance})
        elif fmt == "numpy":
            return "gaussianpdf(x_{var}_, {mean}, {variance})".format(**{'mean': self.mean, 'var': self.var, 'variance': self.variance})
        elif fmt == "mathematica":
            # TODO
            assert False
        elif fmt == "cuda":
            # TODO
            assert False
        else:
            assert False
        

    def smooth_probs(self, alpha, data=None):
        pass

    def eval(self, obs=None):
        if obs is None:
            return

        self.log_der = LOG_ZERO
        self.obs = obs
        
        #self.log_val = logpoissonpmf(math.exp(obs), math.exp(self.mean))
        self.log_val = loggaussianpdf(obs, self.mean, self.variance)
        #print("GAUS", self.log_val, obs, self.mean, self.variance)
        


    def mpe_eval(self, obs):
        return 1.0
  
    def complete(self, data):
        data = numpy.copy(data)
        
        # path = {}
        
        obs = data[self.var]
        if numpy.isnan(obs):
            #obs = floor(self.mean)
            obs = self.mean #observation

            data[self.var] = obs
            # path[self.var] = [self.id]
        
        self.log_mpeval = loggaussianpdf(obs, self.mean, self.variance)
        
        # return (data, path)
        return data
        
        
  
    def n_children(self):
        return 0


    
    def node_type_str(self):
        return GAUSSIAN_VAR_NODE_SYM

    def node_graph_str(self, fnames, astopic=False):
        return "%s%s~Gaus_%s(%s, %s)" % (fnames[self.var], self.var, self.instances, round(self.mean, 3) , round(self.std, 5))

    def node_short_str(self, features=None):
        mean_str = str(self.mean)
        
        vars = self.var
        
        if features is not None:
            vars = features[vars]
        
        return "{type} {id} <{vars}> {mean} {std}".format(type=self.node_type_str(), id=self.id, vars=vars, mean=mean_str, std=self.std)

    def __repr__(self):
        base = Node.__repr__(self)

        return ("""Gaussian Node {line1}
            var: {var} data: [{data}] mean: [{mean}] real_mean: [{rmean}]""".
                format(line1=base,
                       var=self.var,
                       data=self.data.ravel(),
                       mean=self.mean,
                       rmean=numpy.mean(self.data.ravel())))

    def var_values(self):
        return len(self._var_freqs)


class BernoulliNode(Node):

    def __init__(self, data, instances, features):

        Node.__init__(self, data, instances, features)

        self.var = list(self.features)[0]

        self.instances = data.shape[0]
        self.mean = numpy.mean(data[instances, self.var])
        self.variance = numpy.var(data[instances, self.var])


        successones = numpy.sum(numpy.equal(data[instances, self.var], 1))
        successzeros = numpy.sum(numpy.equal(data[instances, self.var], 0))
        if(successones < successzeros):
            _successprob = 1 - (successzeros/len(instances))
        else:
            _successprob = successones/len(instances)
        self.successprob = _successprob
        #assert successones == 0 or successzeros == 0, "succesprob classes error "

        #self.successprob = max(successones,successzeros)/len(instances)
        #print('successprob', self.successprob, successones, successzeros,self.instances, len(instances))
        assert self.variance >= 0

        if self.variance == 0:
            #self.variance = 0.1
            self.variance = 0.0001
        #self.variance = 1

        self._instances = instances
        self.data = data

    def backprop(self):
        # because the gaussian node is both and input node and a node on the bottom layer, we don't need to back propagate to children nodes, there are none
        pass


    def marginalizeToEquation(self, features, evidence, fmt="python", ctx=None):

        if self.var not in features:
            if evidence is None or numpy.isnan(evidence[self.var]):
                return "1.0"

            return "%.30f" % (bernoullipmf(evidence[self.var], self.successprob))

        # return "((e^(-%s)*%s^(x_%s))/x_%s!)"%(round(self.mean,4), round(self.mean,4), self.var, self.var)

        # return "((exp(-%s)*%s**(x_%s))/factorial(x_%s))" % (round(self.mean, 4), round(self.mean, 4), self.var, self.var)

        if fmt == "python":
            return "bernoullipmf(x_{var}_, {successprob})".format(**{'successprob': self.successprob, 'var': self.var})
        elif fmt == "numpy":
            return "bernoullipmf(x_{var}_, {successprob})".format(**{'successprob': self.successprob, 'var': self.var})
        elif fmt == "mathematica":
            # TODO
            assert False
        elif fmt == "cuda":
            # TODO
            assert False
        else:
            assert False

    def smooth_probs(self, alpha, data=None):
        pass

    def eval(self, obs=None):
        if obs is None:
            return

        self.log_der = LOG_ZERO
        self.obs = obs

        self.log_val = logbernoullipmf(obs, self.successprob)
        #print("GAUS", self.log_val, obs, self.mean, self.variance)

    def mpe_eval(self, obs):
        return 1.0

    def complete(self, data):
        data = numpy.copy(data)

        # path = {}

        obs = data[self.var]
        if numpy.isnan(obs):
            #obs = floor(self.mean)
            obs = self.mean #observation

            # TODO check assumptions
            if obs >= 0.5:
                obs = 1
            else:
                obs = 0

            data[self.var] = obs

            # path[self.var] = [self.id]
        #print(self.mean, self.variance)
        self.log_mpeval = logbernoullipmf(obs, self.successprob)
        #self.log_mpeval = loggaussianpdf(obs, self.mean, self.variance)
        # return (data, path)
        return data

    def n_children(self):
        return 0

    def node_type_str(self):
        return BERNOULLI_VAR_NODE_SYM

    def node_graph_str(self, fnames, astopic=False):
        #return "%s%s~Bernoulli_%s(%s)" % (fnames[self.var], self.var, self.instances, self.successprob)
        return "%s~Ber(%s)" % (fnames[self.var], self.successprob)

    def node_short_str(self, features=None):
        mean_str = str(self.mean)

        vars = self.var

        if features is not None:
            vars = features[vars]

        return "{type} {id} <{vars}> {successprob}".format(type=self.node_type_str(), id=self.id, vars=vars, successprob=self.successprob)

    def __repr__(self):
        base = Node.__repr__(self)

        return ("""Bernoulli Node {line1}
            var: {var} data: [{data}] mean: [{mean}] real_mean: [{rmean}]""".
                format(line1=base,
                       var=self.var,
                       data=self.data.ravel(),
                       mean=self.mean,
                       rmean=numpy.mean(self.data.ravel())))

    def var_values(self):
        return len(self._var_freqs)
