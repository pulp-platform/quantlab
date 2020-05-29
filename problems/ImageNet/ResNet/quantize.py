import quantlab.graphs as qg
import quantlab.algorithms as qa


__all__ = ['layers_ste_inq', 'layers_ste_inq_get_controllers']


def layers_ste_inq(logbook, net):
    rule1 = qg.analyse.get_rules_multiple_blocks(['layer{}'.format(k) for k in range(1, 5)])
    rule2 = [qg.analyse.rule_linear_nodes]

    def get_layers_conv_nodes(net):
        net_nodes = qg.analyse.list_nodes(net, verbose=False)
        layers_nodes = qg.analyse.find_nodes(net_nodes, rule1, mix='or')
        conv_nodes = qg.analyse.find_nodes(layers_nodes, rule2, mix='and')
        return conv_nodes

    # add STE in front of convolutions
    quant_ste_config = logbook.config['network']['quantize']['STE']
    conv_nodes = get_layers_conv_nodes(net)
    qg.edit.add_before_linear_ste(net, conv_nodes, num_levels=quant_ste_config['n_levels'], quant_start_epoch=quant_ste_config['quant_start_epoch'])
    # replace convolutions with INQ convolutions
    quant_inq_config = logbook.config['network']['quantize']['INQ']
    conv_nodes = get_layers_conv_nodes(net)
    qg.edit.replace_linear_inq(net, conv_nodes, num_levels=quant_inq_config['n_levels'], quant_init_method=quant_inq_config['quant_init_method'], quant_strategy=quant_inq_config['quant_strategy'])
    return net


def layers_ste_inq_get_controllers(logbook, net):
    net_nodes = qg.analyse.list_nodes(net)
    rule = qg.analyse.get_rules_multiple_blocks(['layer{}'.format(k) for k in range(1, 5)])
    layers_nodes = qg.analyse.find_nodes(net_nodes, rule, mix='or')
    # get STE controller
    quant_ste_ctrl_config = logbook.config['training']['quantize']['STE']
    ste_modules = qa.ste.STEController.get_ste_modules(layers_nodes)
    ste_controller = qa.ste.STEController(ste_modules, quant_ste_ctrl_config['clear_optim_state_on_step'])
    # get INQ controller
    quant_inq_ctrl_config = logbook.config['training']['quantize']['INQ']
    inq_modules = qa.inq.INQController.get_inq_modules(layers_nodes)
    inq_controller = qa.inq.INQController(inq_modules, quant_inq_ctrl_config['schedule'], quant_inq_ctrl_config['clear_optim_state_on_step'])
    return [ste_controller, inq_controller]
