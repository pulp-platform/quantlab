from collections import namedtuple
import torch.nn as nn


__all__ = [
    'Node',
    'list_nodes',
    'find_nodes',
    'rule_linear_nodes',
    'rule_batchnorm_nodes',
    'rule_activation_nodes',
    'rule_dropout_nodes',
    'get_scope_rules',
]


Node = namedtuple('Node', 'name module')


def list_nodes(module, parent_name='', verbose=False):
    """List all the sub-modules of a PyTorch `nn.Module` as named `Node`s."""
    # WARNING: implicit operations in the graph (e.g., reused `nn.ReLU`s) will not be listed!
    # TODO: integrate with explicit graph tracing/scripting ('TorchScript').

    module_nodes = []

    for name, child in module.named_children():
        if len(list(child.children())) == 0:  # leaf PyTorch module
            module_nodes.append(Node(parent_name + name, child))
        else:  # recursive call
            module_nodes.extend(list_nodes(child, parent_name=parent_name + name + '.'))

    if verbose:
        print(module.__class__.__name__)
        print()
        for i, node in enumerate(module_nodes):
            print("{:4d} {:20s} {}".format(i, node.name, node.module))
        print()

    return module_nodes


def find_nodes(nodes, criteria, mix):
    """Find the nodes that satisfy the specified criteria.

    The user should define filtering `criteria` (a list of Python functions)
    to identify a set of nodes in the network. The `criteria` can be mixed in
    two ways:

     * conjunctively (`mix = 'or'`);
     * disjunctively (`mix = 'and'`).
    """

    assert isinstance(criteria, list)

    filtered_nodes = []

    if mix == 'and':
        filtered_nodes.extend(nodes)
        for criterion in criteria:
            filtered_nodes = criterion(filtered_nodes)

    elif mix == 'or':
        for criterion in criteria:
            filtered_nodes.extend(criterion(nodes))

    if len(filtered_nodes) == 0:
        print('No layers satisfying criteria was found!')

    return filtered_nodes


def rule_linear_nodes(nodes):
    """Built-in rule: select nodes performing linear transformations."""
    filtered_nodes = []

    for n in nodes:
        cond1 = n.module.__class__.__name__ in dir(nn.modules.linear) and n.module.__class__.__name__ != 'Identity'
        cond2 = n.module.__class__.__name__ in dir(nn.modules.conv)
        if cond1 or cond2:
            filtered_nodes.append(n)

    return filtered_nodes


def rule_batchnorm_nodes(nodes):
    """Built-in rule: select nodes performing batch-normalisation transformations."""
    filtered_nodes = []

    for n in nodes:
        cond = n.module.__class__.__name__ in dir(nn.modules.batchnorm)
        if cond:
            filtered_nodes.append(n)

    return filtered_nodes


def rule_activation_nodes(nodes):
    """Built-in rule: select nodes performing non-linear transformations."""
    filtered_nodes = []

    for n in nodes:
        cond = n.module.__class__.__name__ in dir(nn.modules.activation)
        if cond:
            filtered_nodes.append(n)

    return filtered_nodes


def rule_dropout_nodes(nodes):
    """Built-in rule: select nodes performing dropout."""
    filtered_nodes = []

    for n in nodes:
        cond = n.module.__class__.__name__ == 'Dropout'
        if cond:
            filtered_nodes.append(n)

    return filtered_nodes


def rule_scope_name(nodes, scope_name):
    """Built-in rule: select nodes by scope name.

    This method should be used in conjunction with informative naming of
    PyTorch network modules.
    """
    filtered_nodes = []

    for n in nodes:
        if n.name.startswith(scope_name):  # I assume that the scope name is a prefix for the node's name
            filtered_nodes.append(n)

    return filtered_nodes


def get_scope_rules(scope_names):
    """Instantiate (possibly multiple) selection rules based on scopes.

    Return `list` of `function`s."""
    if not isinstance(scope_names, list):
        scope_names = [scope_names]

    def scope_rules_generator(scope_name):
        return lambda nodes: rule_scope_name(nodes, scope_name)

    rules = []
    for scope in scope_names:
        rules.append(scope_rules_generator(scope))

    return rules
