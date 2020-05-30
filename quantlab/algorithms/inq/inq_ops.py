# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import math
import itertools
from scipy.optimize import brute, fmin
import torch
import torch.nn as nn

from ..controller import Controller


__all__ = [
    'INQController',
    'INQLinear',
    'INQConv1d',
    'INQConv2d',
]


class INQController(Controller):
    """Control the execution of the INQ quantization algorithm.

    Instantiate typically once per network: provide it with a list of INQ
    modules to control and an INQ schedule; then, insert a call to the `step`
    function once per epoch.
    """
    def __init__(self, modules, schedule, clear_optim_state_on_step=False):#, rescale_weights=False):
        super(INQController, self).__init__()
        self.modules = modules
        self.fraction = 0.0
        self.schedule = {int(k): v for k, v in schedule.items()}  # parse string keys to ints
        self.clear_optim_state_on_step = clear_optim_state_on_step
        # self.rescale_weights = rescale_weights

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k in ('fraction',)}  # parameters not passed at construction time should be stored

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step_pre_training(self, epoch, optimizer=None, tb_writer=None):
        """Call this each epoch before training loop."""
        if epoch in self.schedule.keys():
            self.fraction = self.schedule[epoch]
        else:
            return  # exit immediately: no freezing needed this epoch

        # step each INQ module
        for m in self.modules:
            m.step(self.fraction)
        # clear optimizer state (e.g. Adam's momentum)
        if (optimizer is not None) and self.clear_optim_state_on_step:
            optimizer.state.clear()
        # log INQ fraction to tensorboard
        if tb_writer is not None:
            tb_writer.add_scalar('INQ/fraction', self.fraction, global_step=epoch)

    # def step_post_optim(self, *args, **kwargs):
    #     if self.rescale_weights:
    #         for m in self.modules:
    #             m.weight_inq_ctrl.rescale_weights()
    
    @staticmethod
    def get_inq_modules(nodes_set):
        return [n[1] for n in nodes_set if (isinstance(n[1], INQLinear) or isinstance(n[1], INQConv1d) or isinstance(n[1], INQConv2d))]


class INQNodeController:
    """Used to implement INQ functionality within a custom layer (e.g., INQConv2d).
    Creates and register all relevant fields and parameters in the module. """
    def __init__(self, module, p_name,
                 num_levels=3, quant_init_method=None, quant_strategy="magnitude",  # original 'quant_init_method' was 'uniform-l1-opt'
                 back_compat=True):
        
        self.module = module

        self.p_name = p_name  # `string`
        self.p_name_frozen = None
        self.p_name_s = None  # 's'cale parameter: used to compute quantization levels for parameter 'p'
        if back_compat:
            assert (p_name == 'weight')
            assert (not hasattr(module, 'weight_frozen'))
            assert (not hasattr(module, 'weight_s'))
            self.p_name_frozen = 'weight_frozen'
            self.p_name_s = 'weight_s'
        else:
            # more structured: add support for multiple independent INQ parameters
            self.p_name_frozen = p_name + '_frozen'
            self.p_name_s = p_name + '_s'

        module.__setattr__(self.p_name_frozen,
                           nn.Parameter(torch.full_like(self.weight, float('NaN')),
                                        requires_grad=False))
        module.__setattr__(self.p_name_s,
                           nn.Parameter(torch.full((1,), float('NaN')).to(self.weight),
                                        requires_grad=False))

        self.num_levels = num_levels
        self.quant_init_method = quant_init_method
        self.quant_strategy = quant_strategy  # "magnitude" or "random" or "magnitude-SRQ"/"RPR"
        self.fraction = 0.0

    @property
    def weight(self):
        return self.module.__getattr__(self.p_name)
    
    @property
    def weight_frozen(self):
        return self.module.__getattr__(self.p_name_frozen)
    
    def get_weight_params(self, module):
        weight = module.__getattr__(self.p_name)
        weight_frozen = module.__getattr__(self.p_name_frozen)
        return weight, weight_frozen

    @property
    def s_param(self):
        return self.module.__getattr__(self.p_name_s)
    
    @property
    def s(self):
        return self.s_param[0].item()

    @s.setter
    def s(self, value):
        self.s_param[0] = value

    @staticmethod
    def inq_quantize(weight, quant_levels):
        """Quantize a weight tensor using the INQ quantization scheme."""
        best_quant_level = torch.zeros_like(weight)
        min_quant_err = torch.full_like(weight, float('Inf'))
        for ql in quant_levels:
            quant_err = (weight-ql).abs()
            mask = quant_err < min_quant_err
            best_quant_level[mask] = ql
            min_quant_err[mask] = quant_err[mask]
        quant_weight = best_quant_level
        return quant_weight  # 'min_quant_err' IS NOT RETURNED! DO WE NEED IT?
    
    def inq_step(self, fraction):
        """First, determine quantization levels. Then, quantize given fraction of weights."""

        if self.quant_init_method is None:
            # update s
            if self.fraction == 0.0 and math.isnan(self.s):
                self.s = torch.max(torch.abs(self.weight.data)).item()
            # compute quantization levels
            n_1 = math.floor(math.log((4*self.s)/3, 2))
            n_2 = int(n_1 + 2 - (self.num_levels // 2))
            if self.num_levels >= 3:
                quant_levels_neg = (-2**i for i in range(n_2, n_1+1))
                quant_levels_pos = (2**i for i in range(n_2, n_1+1))
                quant_levels = itertools.chain(quant_levels_neg, [0], quant_levels_pos)
            else: 
                assert(self.num_levels == 2)
                quant_levels = [-self.s/2, self.s/2]  # [-2**n_2, 2**n_2]
        elif self.quant_init_method == 'uniform':
            # update s
            if self.fraction == 0.0 and math.isnan(self.s):
                self.s = torch.max(torch.abs(self.weight.data)).item()
            # compute quantization levels
            quant_levels = torch.linspace(-self.s, self.s, steps=self.num_levels)
        elif self.quant_init_method in ['uniform-l1-opt',
                                        'uniform-l2-opt',
                                        'uniform-l2-opt-per_ch',
                                        'uniform-linf-opt']:
            get_quant_levels = lambda s: torch.linspace(-s, s, steps=self.num_levels).to(self.weight)
            if self.fraction == 0.0 and math.isnan(self.s):
                def optim_weight(weight):
                    def loss(s):
                        s = s.item()
                        quant_levels = get_quant_levels(s)
                        min_quant_err = torch.full_like(weight, float('Inf'))
                        for i, ql in enumerate(quant_levels):
                            tmp = (weight-ql).abs()
                            min_quant_err = torch.min(min_quant_err, tmp)
                        if self.quant_init_method == 'uniform-l1-opt':
                            return min_quant_err.norm(p=1).item()
                        elif self.quant_init_method in ['uniform-l2-opt', 'uniform-l2-opt-per_ch']:
                            return min_quant_err.norm(p=2).item()
                        elif self.quant_init_method == 'uniform-linf-opt':
                            return min_quant_err.norm(p=float('Inf')).item()
                        else:
                            assert(False)
                    bounds = (1e-6, weight.abs().max().item())
                    opt_res = brute(loss, ranges=(bounds,),
                                    Ns=1000, disp=True,
                                    finish=fmin)
                    s = opt_res[0]
                    weight.mul_(1/s)
                    s = 1
                    return s  # 'optim_weight' HAS THE SIDE-EFFECT OF CHANGING 'weight'
                              # WOULD IT NOT BE BETTER REMOVING OUTPUT 's' AND ENSURING STATE CONSISTENCY?
                
                if self.quant_init_method in ['uniform-l1-opt',
                                              'uniform-l2-opt',
                                              'uniform-linf-opt']:
                    self.s = optim_weight(self.weight.data.flatten().detach())
                elif self.quant_init_method in ['uniform-l2-opt-per_ch']:
                    self.s = 1
                    for c in range(self.weight.size(0)):
                        optim_weight(self.weight.data[c].flatten().detach())
            quant_levels = get_quant_levels(self.s)
        else:
            assert False

        self.fraction = fraction

        if self.quant_strategy == "magnitude-SRQ" or self.quant_strategy == "RPR":
            if self.fraction is None:
                return
            # get current weights quantized
            self.weight_frozen.data.copy_(self.inq_quantize(self.weight.data, quant_levels))
            num_unfreeze = int((1-self.fraction) * self.weight.numel())
            idx_unfreeze = torch.randperm(self.weight.numel())[:num_unfreeze]
            self.weight_frozen.data.flatten()[idx_unfreeze] = float('NaN')
        elif self.quant_strategy == "magnitude" or self.quant_strategy == "random":
            # get number of weights to quantize
            n_weights = self.weight_frozen.numel()
            old_count = n_weights - torch.isnan(self.weight_frozen.data).sum(dtype=torch.long).item()
            new_count = int(self.fraction * n_weights)
            # find indexes of weights to quantize
            if self.quant_strategy == "magnitude":
                idx_full_prec = torch.nonzero(torch.isnan(self.weight_frozen.flatten()))
                if idx_full_prec.numel() > 0:
                    idx_full_prec = idx_full_prec[:, 0]
                _, idx = self.weight.flatten()[idx_full_prec].abs().sort(descending=True)
                idx_new_quant = idx_full_prec[idx[:new_count-old_count]]
            elif self.quant_strategy == "random":
                idx_new_quant = torch.nonzero(torch.isnan(self.weight_frozen.flatten()))[torch.randperm(new_count-old_count)]
            else:
                assert False
            idx_freeze = idx_new_quant[:new_count-old_count]
            # quantize the weights at these indexes
            self.weight_frozen.data.flatten()[idx_freeze] = self.inq_quantize(self.weight.data.flatten()[idx_freeze], quant_levels)
        else:
            assert False
    
    def inq_assemble_weight(self, module=None):
        
        # with nn.DataParallel, the module is copied, so self.module cannot be used
        # WAS THE FUNCTION 'get_weight_params' DESIGNED LATER THAN '@property's 'weight' AND 'weigth_frozen'?
        weight, weight_frozen = self.get_weight_params(module)
        
        weight_frozen = weight_frozen.detach()
        idx_frozen = ~torch.isnan(weight_frozen)
        weight_assembled = torch.zeros_like(weight_frozen)
        weight_assembled[idx_frozen] = weight_frozen[idx_frozen]  # quantized part
        mask_full_prec = torch.isnan(weight_frozen).float()
        weight_full_prec = mask_full_prec * weight
        weight_assembled = weight_assembled + weight_full_prec  # weight_frozen[frozen] + weight[~frozen]
        return weight_assembled
    
    def rescale_weights(self):
        eps = 1e-8  # HARD CODED: MAYBE '__eps__' IS BETTER IDENTIFIER
        self.weight.data.mul_((self.s / 2) / (self.weight.data.abs().mean().item() + eps))


# INQNodeController's '__init__' and 'inq_assemble_weight' methods take a module as input
# in particular, 'inq_assemble_weight' uses an helper function to assist with 'nn.DataParallel'
# see comment on line 232
class INQLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, 
                 num_levels=3, quant_init_method=None, quant_strategy="magnitude"):
        
        super(INQLinear, self).__init__(in_features, out_features, bias)
        self.weight_inq_ctrl = INQNodeController(self, 'weight',
                                                 num_levels=num_levels,
                                                 quant_init_method=quant_init_method,
                                                 quant_strategy=quant_strategy)
    
    def step(self, fraction):
        self.weight_inq_ctrl.inq_step(fraction)

    def forward(self, input):
        weight_assembled = self.weight_inq_ctrl.inq_assemble_weight(self)
        return nn.functional.linear(input, weight_assembled, self.bias)
    
    
class INQConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, 
                 bias=True, padding_mode='zeros', 
                 num_levels=3, quant_init_method=None, quant_strategy="magnitude"):
        
        super(INQConv1d, self).__init__(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, 
                 bias, padding_mode)
        
        self.weight_inq_ctrl = INQNodeController(self, 'weight',
                                                 num_levels=num_levels,
                                                 quant_init_method=quant_init_method,
                                                 quant_strategy=quant_strategy)
        
    def step(self, fraction):
        self.weight_inq_ctrl.inq_step(fraction)

    def forward(self, input):
        weight_assembled = self.weight_inq_ctrl.inq_assemble_weight(self)
        
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return nn.functional.conv1d(
                    nn.functional.pad(input, expanded_padding, mode='circular'),
                    weight_assembled, self.bias, self.stride,
                    (0,), self.dilation, self.groups)
        return nn.functional.conv1d(input, weight_assembled, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
        
    
class INQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, 
                 bias=True, #padding_mode='zeros', 
                 num_levels=3, quant_init_method=None, quant_strategy="magnitude"):
        
        super(INQConv2d, self).__init__(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, 
                 bias)#, padding_mode) # removed padding_mode for backward comp. to 0.4.1
        
        self.weight_inq_ctrl = INQNodeController(self, 'weight',
                                                 num_levels=num_levels,
                                                 quant_init_method=quant_init_method,
                                                 quant_strategy=quant_strategy)
        
    def step(self, fraction):
        self.weight_inq_ctrl.inq_step(fraction)

    def forward(self, input):
        weight_assembled = self.weight_inq_ctrl.inq_assemble_weight(self)
        
#        if self.padding_mode == 'circular':
#            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
#                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
#            return nn.functional.conv2d(nn.functional.pad(input, expanded_padding, mode='circular'),
#                                        weightAssembled, self.bias, self.stride,
#                                        (0,), self.dilation, self.groups)

        return nn.functional.conv2d(input, weight_assembled, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)


# if __name__ == '__main__':
#     x = torch.linspace(-2,2,100)
#     num_levels = 3
#     s = torch.max(torch.abs(x)).item()
#
#     n_1 = math.floor(math.log((4*s)/3, 2))
#     n_2 = int(n_1 + 2 - (num_levels//2))
#     quant_levels_pos = (2**i for i in range(n_2, n_1+1))
#     quant_levels_neg = (-2**i for i in range(n_2, n_1+1))
#     quant_levels = itertools.chain(quant_levels_pos, [0], quant_levels_neg)
#
#     x_q = INQNodeController.inq_quantize(x, quant_levels)
#
#
#     import matplotlib.pyplot as plt
#     plt.clf()
#     plt.plot(x.numpy())
#     plt.plot(x_q.numpy())
#
#
#     model = INQLinear(2, 3, bias=False,
#                       num_levels=num_levels, quant_strategy="RPR")
#
#     print(model.weight)
#     print(model.weight_frozen)
#     model.step(0.5)
#     print(model.weight)
#     print(model.weight_frozen)
#
#     x = torch.randn(4,2)
#     y = model(x)
#     L = y.norm(p=2)
#     L.backward()
