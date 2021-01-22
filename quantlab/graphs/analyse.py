from collections import namedtuple
import torch.nn as nn


__all__ = [
    'list_nodes',
    'find_nodes',
    'rule_linear_nodes',
    'rule_batchnorm_nodes',
    'rule_activation_nodes',
    'get_rules_multiple_blocks',
    'Node'
]


Node = namedtuple('Node', 'name module')


def list_nodes(net, name='', verbose=False):
    """Take a PyTorch `nn.Module` and return a `list` of `Node`s.

    WARNING: implicit operations in the graph (e.g., reused `nn.ReLU`s) will
    not be listed!
    """
    # TODO: integrate with explicit graph tracing/scripting ('TorchScript').
    net_nodes = []
    for n, m in net.named_children():
        if list(m.children()) == []:
            node = Node(name+n, m)
            net_nodes.append(node)
            if verbose:
                print(node)
        else:
            m_nodes = list_nodes(m, name=name+n+'.')
            net_nodes.extend(m_nodes)
    return net_nodes


def find_nodes(net_nodes, criteria, mix):
    """Find the set of nodes that satisfy user-defined conditions.

    The user should define ''filtering'' `criteria` (a list of Python
    functions) to identify a set of nodes in the network. Later, these nodes
    could be replaced or complemented to support a specified quantization
    algorithm.
    The `criteria` can be mixed in two ways:
     * conjunctively (`mix = 'or'`);
     * disjunctively (`mix = 'and'`).

    Return `list` of `Node`s."""
    nodes_set = []
    if mix == 'and':
        nodes_set.extend(net_nodes)
        for crit in criteria:
            nodes_set = crit(nodes_set)
    elif mix == 'or':
        for crit in criteria:
            nodes_set.extend(crit(net_nodes))
    if len(nodes_set) == 0:
        print('No layers satisfying criteria was found!')
    return nodes_set


def rule_linear_nodes(nodes_set):
    """Example built-in rule: select nodes performing linear transformations.

    Return `list` of `Node`s."""
    filtered_nodes_set = []
    for n, m in nodes_set:
        cond1 = m.__class__.__name__ in dir(nn.modules.linear) and m.__class__.__name__ != 'Identity'
        cond2 = m.__class__.__name__ in dir(nn.modules.conv)
        if cond1 or cond2:
            filtered_nodes_set.append(Node(n, m))
    return filtered_nodes_set


def rule_batchnorm_nodes(nodes_set):
    """Example built-in rule: select nodes performing batch-normalisation transformations.

    Return `list` fo `Node`s."""
    filtered_nodes_set = []
    for n, m in nodes_set:
        cond = m.__class__.__name__ in dir(nn.modules.batchnorm)
        if cond:
            filtered_nodes_set.append(Node(n, m))
    return filtered_nodes_set


def rule_activation_nodes(nodes_set):
    """Example built-in rule: select nodes performing non-linear transformations.

    Return `list` of `Node`s."""
    filtered_nodes_set = []
    for n, m in nodes_set:
        cond = m.__class__.__name__ in dir(nn.modules.activation)
        if cond:
            filtered_nodes_set.append(Node(n, m))
    return filtered_nodes_set


def rule_single_block(nodes_set, block_name):
    """Example built-in rule: select nodes by block (name prefix).

    Return `list` of `Node`s."""
    filtered_nodes_set = []
    for n, m in nodes_set:
        if n.startswith(block_name):  # block name is a prefix of node's name
            filtered_nodes_set.append(Node(n, m))
    return filtered_nodes_set


def get_rules_multiple_blocks(block_names):
    """Use lambda expressions and currying to instantiate multiple block selection rules.

    Return `list` of `Node`s."""
    blocks_selector_rules = lambda block_name: lambda nodes_set: rule_single_block(nodes_set, block_name)
    rules = []
    for bn in block_names:
        rules.append(blocks_selector_rules(bn))
    return rules
