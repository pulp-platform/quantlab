import quantlib.graphs as qg
import quantlib.algorithms as qa


__all__ = ['layers_ste_inq', 'layers_ste_inq_get_controllers']


def layers_ste_inq(config, net):

    def get_layers_conv_nodes(net):

        net_nodes = qg.analyse.list_nodes(net, verbose=False)

        rule1 = qg.analyse.get_rules_multiple_blocks(['layer{}'.format(k) for k in range(1, 5)])
        layers_nodes = qg.analyse.find_nodes(net_nodes, rule1, mix='or')

        rule2 = [qg.analyse.rule_linear_nodes]
        conv_nodes = qg.analyse.find_nodes(layers_nodes, rule2, mix='and')

        return conv_nodes

    # add STE in front of convolutions
    ste_config = config['STE']
    conv_nodes = get_layers_conv_nodes(net)
    qg.edit.add_before_linear_ste(net, conv_nodes, num_levels=ste_config['n_levels'], quant_start_epoch=ste_config['quant_start_epoch'])

    # replace convolutions with INQ convolutions
    inq_config = config['INQ']
    conv_nodes = get_layers_conv_nodes(net)
    qg.edit.replace_linear_inq(net, conv_nodes, num_levels=inq_config['n_levels'], quant_init_method=inq_config['quant_init_method'], quant_strategy=inq_config['quant_strategy'])

    return net


def layers_ste_inq_get_controllers(config, net):

    net_nodes = qg.analyse.list_nodes(net)
    rule = qg.analyse.get_rules_multiple_blocks(['layer{}'.format(k) for k in range(1, 5)])
    layers_nodes = qg.analyse.find_nodes(net_nodes, rule, mix='or')

    # get STE controller
    ste_ctrl_config = config['STE']
    ste_modules = qa.ste.STEController.get_ste_modules(layers_nodes)
    ste_controller = qa.ste.STEController(ste_modules, ste_ctrl_config['clear_optim_state_on_step'])

    # get INQ controller
    inq_ctrl_config = config['INQ']
    inq_modules = qa.inq.INQController.get_inq_modules(layers_nodes)
    inq_controller = qa.inq.INQController(inq_modules, inq_ctrl_config['schedule'], inq_ctrl_config['clear_optim_state_on_step'])

    return [ste_controller, inq_controller]
