# Copyright (c) 2019 ETH Zurich, Lukas Cavigelli

import torch

from ..controller import Controller


__all__ = [
    'STEController',
    'STEActivation',
]


class STEController(Controller):
    def __init__(self, modules, clear_optim_state_on_step=False):
        super(STEController).__init__()
        self.modules = modules
        self.clear_optim_state_on_step = clear_optim_state_on_step

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k in ()}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step_pre_training(self, epoch, optimizer=None, tb_writer=None):
        # step each STE module
        for m in self.modules:
            m.step(epoch)
        if (optimizer is not None) and self.clear_optim_state_on_step:
            for m in self.modules:
                if m.quant_start_epoch == epoch:
                    optimizer.state.clear()  # weight decay?

    @staticmethod
    def get_ste_modules(nodes_set):
        return [n[1] for n in nodes_set if isinstance(n[1], STEActivation)]


class STEActivation(torch.nn.Module):
    """Quantizes activations according to the straight-through estiamtor (STE).
    Needs a STEController, if `quant_start_epoch` > 0.

    monitor_epoch: In this epoch, keep track of the maximal activation value (absolute value).
        Then (at epoch >= quant_start_epoch), clamp the values to [-max, max], and then do quantization.
        If monitor_epoch is None, max=1 is used."""
    def __init__(self, num_levels=2**8-1, quant_start_epoch=0):
        super(STEActivation, self).__init__()
        assert(num_levels >= 2)
        self.num_levels = num_levels
        self.abs_max_value = torch.nn.Parameter(torch.ones(1), requires_grad=False)

        assert(quant_start_epoch >= 0)
        self.quant_start_epoch = quant_start_epoch
        self.started = self.quant_start_epoch == 0

        self.monitor_epoch = self.quant_start_epoch - 1
        self.monitoring = False
        if 0 <= self.monitor_epoch:
            self.monitoring = self.monitor_epoch == 0

    def step(self, epoch):
        if self.monitor_epoch == epoch:
            self.monitoring = True
            self.abs_max_value.data[0] = 0.0  # prepare to log maximum activation value
        else:
            self.monitoring = False

        if self.quant_start_epoch <= epoch:
            self.started = True

    @staticmethod
    def ste_round_functional(x):
        return x - (x - x.round()).detach()

    def forward(self, x):
        if self.monitoring:
            self.abs_max_value.data[0] = max(self.abs_max_value.item(), x.abs().max())
            
        if self.started:
            x = x / self.abs_max_value.item()  # map from [-max, max] to [-1, 1]
            xclamp = x.clamp(-1, 1)
            y = xclamp
            y = (y + 1) / 2  # map from [-1,1] to [0,1]
            y = STEActivation.ste_round_functional(y * (self.num_levels - 1)) / (self.num_levels - 1)
            y = 2 * y - 1  # map from [0, 1] to [-1, 1]
            y = y * self.abs_max_value.item()  # map from [-1, 1] to [-max, max]
        else:
            y = x

        return y


# if __name__ == "__main__":
#     u = torch.randn(10, requires_grad=True)
#     x = u * 2
#     y = STEActivation(num_levels=2)(x)
#     L = y.norm(2)  # pull to 0
#     L.backward()
