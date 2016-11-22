


from collections import deque
import itertools
import logging
from math import ceil
import random

import numpy
import scipy.sparse
import sklearn.preprocessing

from spn import INT_TYPE
from spn.linked.layers import PoissonLayer, GaussianLayer, BernoulliLayer
from spn.linked.layers import ProductLayer as ProductLayerLinked
from spn.linked.layers import SumLayer as SumLayerLinked
from spn.linked.nodes import ProductNode, GaussianNode, BernoulliNode
from spn.linked.nodes import SumNode, PoissonNode
from spn.linked.spn import Spn as SpnLinked
from spn.utils import pairwise


class SpnFactory(object):

    @classmethod
    def layered_linked_spn(cls, root_node, data, config={}):
        """
        Given a simple linked version (parent->children),
        returns a layered one (linked + layers)
        """
        layers = []
        root_layer = None
        input_nodes = []
        layer_nodes = []
        input_layer = None

        # layers.append(root_layer)
        previous_level = None

        # collecting nodes to visit
        open = deque()
        next_open = deque()
        closed = set()

        open.append(root_node)

        while open:
            # getting a node
            current_node = open.popleft()
            current_id = current_node.id

            # has this already been seen?
            if current_id not in closed:
                closed.add(current_id)
                layer_nodes.append(current_node)
                # print('CURRENT NODE')
                # print(current_node)

                # expand it
                for child in current_node.children:
                    # only for non leaf nodes
                    if (isinstance(child, SumNode) or
                            isinstance(child, ProductNode)):
                        next_open.append(child)
                    else:
                        # it must be an input node
                        if child.id not in closed:
                            input_nodes.append(child)
                            closed.add(child.id)

            # open is now empty, but new open not
            if (not open):
                # swap them
                open = next_open
                next_open = deque()

                # and create a new level alternating type
                if previous_level is None:
                    # it is the first level
                    if isinstance(root_node, SumNode):
                        previous_level = SumLayerLinked([root_node])
                    elif isinstance(root_node, ProductNode):
                        previous_level = ProductLayerLinked([root_node])
                elif isinstance(previous_level, SumLayerLinked):
                    previous_level = ProductLayerLinked(layer_nodes)
                elif isinstance(previous_level, ProductLayerLinked):
                    previous_level = SumLayerLinked(layer_nodes)

                layer_nodes = []

                layers.append(previous_level)

        if isinstance(input_nodes[0], PoissonNode):
            input_layer = PoissonLayer(input_nodes)
        if isinstance(input_nodes[0], GaussianNode):
            input_layer = GaussianLayer(input_nodes)
        if isinstance(input_nodes[0], BernoulliNode):
            input_layer = BernoulliLayer(input_nodes)

        spn = SpnLinked(data, input_layer=input_layer, layers=layers[::-1], config=config)
        return spn

    @classmethod
    def pruned_spn_from_slices(cls, node_assoc, building_stack, prune=True, logger=None):
        """
        WRITEME
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        # traversing the building stack
        # to link and prune nodes
        for build_node in reversed(building_stack):

            # current node
            current_id = build_node.id
            # print('+ Current node: %d', current_id)
            current_children_slices = build_node.children
            # print('\tchildren: %r', current_children_slices)
            current_children_weights = build_node.weights
            # print('\tweights: %r', current_children_weights)

            # retrieving corresponding node
            node = node_assoc[current_id]
            # print('retrieved node', node)

            # discriminate by type
            if isinstance(node, SumNode):
                logging.debug('it is a sum node %d', current_id)
                # getting children
                for child_slice, child_weight in zip(current_children_slices,
                                                     current_children_weights):
                    # print(child_slice)
                    # print(child_slice.id)
                    # print(node_assoc)
                    child_id = child_slice.id
                    child_node = node_assoc[child_id]
                    # print(child_node)

                    # checking children types as well
                    if isinstance(child_node, SumNode) and prune:
                        logging.debug('++ pruning node: %d', child_node.id)
                        # this shall be pruned
                        for grand_child, grand_child_weight \
                                in zip(child_node.children,
                                       child_node.weights):
                            node.add_child(grand_child,
                                           grand_child_weight * 
                                           child_weight)

                    else:
                        logging.debug('+++ Adding it as child: %d',
                                      child_node.id)
                        node.add_child(child_node, child_weight)
                        # print('children added')

            elif isinstance(node, ProductNode):
                logging.debug('it is a product node %d', current_id)
                # linking children
                for child_slice in current_children_slices:
                    child_id = child_slice.id
                    child_node = node_assoc[child_id]

                    # checking for alternating type
                    if isinstance(child_node, ProductNode) and prune:
                        logging.debug('++ pruning node: %d', child_node.id)
                        # this shall be pruned
                        for grand_child in child_node.children:
                            node.add_child(grand_child)
                    else:
                        node.add_child(child_node)
                        # print('+++ Linking child %d', child_node.id)

        # this is superfluous, returning a pointer to the root
        root_build_node = building_stack[0]
        return node_assoc[root_build_node.id]

    @classmethod
    def layered_pruned_linked_spn(cls, root_node):
        """
        WRITEME
        """
        #
        # first traverse the spn top down  to collect a bottom up traversal order
        # it could be done in a single pass I suppose, btw...
        building_queue = deque()
        traversal_stack = deque()

        building_queue.append(root_node)

        while building_queue:
            #
            # getting current node
            curr_node = building_queue.popleft()
            #
            # appending it to the stack
            traversal_stack.append(curr_node)
            #
            # considering children
            try:
                for child in curr_node.children:
                    building_queue.append(child)
            except:
                pass
        #
        # now using the inverse traversal order
        for node in reversed(traversal_stack):

            # print('retrieved node', node)

            # discriminate by type
            if isinstance(node, SumNode):

                logging.debug('it is a sum node %d', node.id)
                current_children = node.children[:]
                current_weights = node.weights[:]

                # getting children
                children_to_add = deque()
                children_weights_to_add = deque()
                for child_node, child_weight in zip(current_children,
                                                    current_weights):
                    # print(child_slice)
                    # print(child_slice.id)
                    # print(node_assoc)

                    print(child_node)

                    # checking children types as well
                    if isinstance(child_node, SumNode):
                        # this shall be prune
                        logging.debug('++ pruning node: %d', child_node.id)
                        # del node.children[i]
                        # del node.weights[i]

                        # adding subchildren
                        for grand_child, grand_child_weight \
                                in zip(child_node.children,
                                       child_node.weights):
                            children_to_add.append(grand_child)
                            children_weights_to_add.append(grand_child_weight * 
                                                           child_weight)
                            # node.add_child(grand_child,
                            #                grand_child_weight *
                            #                child_weight)

                        # print(
                        #     'remaining  children', [c.id for c in node.children])
                    else:
                        children_to_add.append(child_node)
                        children_weights_to_add.append(child_weight)

                #
                # adding all the children (ex grand children)
                node.children.clear()
                node.weights.clear()
                for child_to_add, weight_to_add in zip(children_to_add, children_weights_to_add):
                    node.add_child(child_to_add, weight_to_add)

                    # else:
                    #     print('+++ Adding it as child: %d', child_node.id)
                    #     node.add_child(child_node, child_weight)
                    #     print('children added')

            elif isinstance(node, ProductNode):

                logging.debug('it is a product node %d', node.id)
                current_children = node.children[:]

                children_to_add = deque()
                # linking children
                for i, child_node in enumerate(current_children):

                    # checking for alternating type
                    if isinstance(child_node, ProductNode):

                        # this shall be pruned
                        logging.debug('++ pruning node: %d', child_node.id)
                        # this must now be useless
                        # del node.children[i]

                        # adding children
                        for grand_child in child_node.children:
                            children_to_add.append(grand_child)
                            # node.add_child(grand_child)
                    else:
                        children_to_add.append(child_node)
                    #     node.add_child(child_node)
                    #     print('+++ Linking child %d', child_node.id)
                #
                # adding grand children
                node.children.clear()
                for child_to_add in children_to_add:
                    node.add_child(child_to_add)


        #
        # now transforming it layer wise
        # spn = SpnFactory.layered_linked_spn(root_node)
        return root_node


