# DESCRIPTION
#
# This recipe will quantize just the layers under the 'features' scope.
# In particular, it will apply the following algorithms:
#   - STE to activation functions;
#   - INQ to convolutional/linear layers.
#
# There will be just two controllers:
#   - one for all the STE modules;
#   - one for all the INQ modules.
#

import quantlib.graphs as qg
import quantlib.algorithms as qa


__all__ = ['quantize', 'get_controllers']


def quantize(config, net):

    def get_features_convlinear_nodes(net):

        net_nodes = qg.analyse.list_nodes(net, verbose=False)

        rule1 = qg.analyse.get_scope_rules('features')
        features_nodes = qg.analyse.find_nodes(net_nodes, rule1, mix='or')

        rule2 = [qg.analyse.rule_linear_nodes]
        convlinear_nodes = qg.analyse.find_nodes(features_nodes, rule2, mix='and')

        return convlinear_nodes

    # add STE in front of convolutions
    ste_config = config['params']['STE']
    convlinear_nodes = get_features_convlinear_nodes(net)
    qg.edit.add_before_linear_ste(net, convlinear_nodes, num_levels=ste_config['n_levels'], quant_start_epoch=ste_config['quant_start_epoch'])

    # replace convolutions with INQ convolutions
    inq_config = config['params']['INQ']
    conv_nodes = get_features_convlinear_nodes(net)
    qg.edit.replace_linear_inq(net, conv_nodes, num_levels=inq_config['n_levels'], quant_init_method=inq_config['quant_init_method'], quant_strategy=inq_config['quant_strategy'])

    return net


def get_controllers(config, net):

    net_nodes = qg.analyse.list_nodes(net)
    rule = qg.analyse.get_scope_rules('features')
    features_nodes = qg.analyse.find_nodes(net_nodes, rule, mix='or')

    # get STE controller
    ste_ctrl_config = config['STE']
    ste_modules = qa.ste.STEController.get_ste_modules(features_nodes)
    ste_controller = qa.ste.STEController(ste_modules, ste_ctrl_config['clear_optim_state_on_step'])

    # get INQ controller
    inq_ctrl_config = config['INQ']
    inq_modules = qa.inq.INQController.get_inq_modules(features_nodes)
    inq_controller = qa.inq.INQController(inq_modules, inq_ctrl_config['schedule'], inq_ctrl_config['clear_optim_state_on_step'])

    return [ste_controller, inq_controller]