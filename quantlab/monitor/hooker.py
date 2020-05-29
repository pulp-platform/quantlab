import torch
from functools import partial

# PyTorch hooks are functions which must have signature
#
#       IN:     (module, input, output)
#       OUT:    None
#
# In general, one would like to apply a different operation to `input` and/or `output`
# depending on the layer of the network where the hook will be called.
# This can be achieved by "currying" a prototypical hook with a function `f_module`:
# in lambda calculus terms, the prototypical hook can be written as a lambda expression
#
#       g = \lambda f. f (m i o)
#
# where m, i, o are "free variables" and f in the "bound variable".
# We can apply `g` to `f_module` to obtain the lambda expression
#
#       f_module (m i o)
#
# i.e., the desired module-specific hook.
# In Python, this can be achieved by treating `g` as a function of four variables
# (f, m, i, o) to which a function argument `f_module` is passed and which returns
# a function `f_module` with signature
#
#       IN:     (m, i, o)
#       OUT:    None
#
# This is exactly what can be achieved using the `partial` function provided by
# the `functools` module.
#

# Credit to:
#   Software design: https://www.youtube.com/watch?v=yk-IXz0DjTY
#   Tracking functionalities: https://cs231n.github.io/neural-networks-3/
#


__all__ = [
    'WeightUpdateTracker',
    'LinearOutputTracker',
    'GammaBetaTracker',
    'BatchNormInputTracker',
    'ActivationInputTracker',
]


# module/step-specific logger functions
def tb_scalar_logger_generator(writer):
    return lambda kwargs: lambda global_step: lambda tag: lambda value: writer.add_scalar(tag, value, global_step=global_step, **kwargs)


def tb_histogram_logger_generator(writer):
    return lambda kwargs: lambda global_step: lambda tag: lambda value: writer.add_histogram(tag, value, global_step=global_step, **kwargs)


# forward hooks prototypes
__FORWARD_HOOKS__ = [
    'protohook_fw_out_max',
    'protohook_fw_out_min',
    'protohook_fw_out_dist',
    'protohook_fw_in_dist',
    'protohook_fw_in_per_ch_mean_dist',
    'protohook_fw_in_per_ch_std_dist',
    # 'protohook_fw_in_per_ch_std_min',
]


def protohook_fw_out_max(writer, module, input, output):
    stat = torch.max(output)
    writer(stat)


def protohook_fw_out_min(writer, module, input, output):
    stat = torch.min(output)
    writer(stat)


def protohook_fw_out_dist(writer, module, input, output):
    stat = output
    writer(stat)


def protohook_fw_in_dist(writer, module, input, output):
    stat = input[0]
    writer(stat)


def protohook_fw_in_per_ch_mean_dist(writer, module, input, output):
    bs = input[0].shape[0]
    nc = input[0].shape[1]
    temp = input[0].view(bs, nc, -1)
    stat = temp.sum(-1).sum(0) / (temp.shape[0] * temp.shape[-1])
    writer(stat)


def protohook_fw_in_per_ch_std_dist(writer, module, input, output):
    bs = input[0].shape[0]
    nc = input[0].shape[1]
    temp = input[0].view(bs, nc, -1)
    mean = temp.sum(-1).sum(0) / (temp.shape[0] * temp.shape[-1])
    sqrd = (temp - mean[:, None])**2
    stat = torch.sqrt(sqrd.sum(-1).sum(0) / (temp.shape[0] * temp.shape[-1] - 1))
    writer(stat)


# def protohook_fw_in_per_ch_std_min(writer, module, input, output):
#     bs = input[0].shape[0]
#     nc = input[0].shape[1]
#     temp = input[0].view(bs, nc, -1)
#     mean = temp.sum(-1).sum(0) / (temp.shape[0] * temp.shape[-1])
#     sqrd = (temp - mean[:, None])**2
#     stat = torch.min(torch.sqrt(sqrd.sum(-1).sum(0) / (temp.shape[0] * temp.shape[-1] - 1)))
#     writer(stat)


# backward hooks prototypes
__BACKWARD_HOOKS__ = [
    'protohook_bw_out_max',
    'protohook_bw_out_min',
    'protohook_bw_out_dist',
    'protohook_bw_in_dist',
    'protohook_bw_out_per_ch_norm_dist',
]


def protohook_bw_out_max(writer, module, input, output):
    stat = torch.max(output[0])
    writer(stat)


def protohook_bw_out_min(writer, module, input, output):
    stat = torch.min(output[0])
    writer(stat)


def protohook_bw_out_dist(writer, module, input, output):
    stat = output[0]
    writer(stat)


def protohook_bw_in_dist(writer, module, input, output):
    if input[0] is not None:
        stat = input[0]
        writer(stat)


def protohook_bw_out_per_ch_norm_dist(writer, module, input, output):
    bs = output[0].shape[0]
    nc = output[0].shape[1]
    temp = output[0].view(bs, nc, -1)
    stat = torch.norm(temp, 2, dim=-1).mean(0)
    writer(stat)


# prototype tensor tracker
class _TensorTracker(object):

    def __init__(self, writer, **kwargs):
        self.writer = writer

    def setup(self, net, step):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError


# parameters tracker
class WeightUpdateTracker(_TensorTracker):
    def __init__(self, writer, modules, **kwargs):
        super(WeightUpdateTracker, self).__init__(writer)
        self.tag = 'Linear'
        self.modules = {k: None for k in modules}
        self.writer_gen_k = tb_scalar_logger_generator(self.writer)(kwargs)
        self.writer_gen_k_e = None
        self.net = None

    def setup(self, net, step):
        if self.net is None:
            self.net = net
        self.writer_gen_k_e = self.writer_gen_k(step)
        for n, m in self.net.named_modules():
            if n in self.modules.keys():
                self.modules[n] = m.weight.clone().detach().cpu()
                stat = torch.norm(self.modules[n])  # weight tensor norm
                writer_gen_k_e_t = self.writer_gen_k_e(self.tag+'_weight_norm/'+ntag)
                writer_gen_k_e_t(stat)

    def release(self):
        for n, m in self.net.named_modules():
            if n in self.modules.keys():
                w_old_norm = torch.norm(self.modules[n])
                w_new = m.weight.clone().detach().cpu()
                w_upd = w_new - self.modules[n]
                stat = torch.norm(w_upd) / w_old_norm  # update vector norm/weight vector norm ratio
                writer_gen_k_e_t = self.writer_gen_k_e(self.tag+'_update_norm_weight_norm_ratio/'+ntag)
                writer_gen_k_e_t(stat)
                self.modules[n] = None  # release weight tensor copy to garbage collector


class GammaBetaTracker(_TensorTracker):
    def __init__(self, writer, modules, **kwargs):
        super(GammaBetaTracker, self).__init__(writer)
        self.tag = 'Batch-normalisation'
        self.modules = {k: None for k in modules}
        self.writer_gen_k = tb_histogram_logger_generator(self.writer)(kwargs)
        self.writer_gen_k_e = None
        self.net = None

    def setup(self, net, step):
        if self.net is None:
            self.net = net
        self.writer_gen_k_e = self.writer_gen_k(step)
        for n, m in self.net.named_modules():
            if n in self.modules.keys():
                stat = m.weight
                writer_gen_k_e_t = self.writer_gen_k_e(self.tag+'_parameters/Gamma/'+ntag)
                writer_gen_k_e_t(stat)
                stat = m.bias
                writer_gen_k_e_t = self.writer_gen_k_e(self.tag+'_parameters/Beta/'+ntag)
                writer_gen_k_e_t(stat)

    def release(self):
        pass


# "internal" tensors trackers
class _HookTracker(_TensorTracker):

    def __init__(self, writer, tag, modules, hooks_generators):
        """Track statistics at the back- or front-interface of given modules.

        Usually, no symbolic handles (i.e., Python identifiers) are available
        for the tensors processed by those PyTorch `nn.Module`s implementing
        the hidden layers of the represented deep neural network; therefore,
        tracking statistics about these tensors requires to use "hooks" placed
        at either the back- or front-interface of the specific target module.

        Given a specific `nn.Module` we refer to its back-interface as the set
        of tensors taken as inputs during the forward pass and of tensors
        produced as outputs during the backward pass (usually gradients); we
        refer to the front-interface as the set of tensors produced as outputs
        during the forward pass and of tensors taken as inputs during the
        backward pass (usually gradients).

        The user who wants to monitor some statistics of the tensors in the
        back- or the front-interface of a list of `Node`s should provide the
        TensorBoard `writer` object, a `tag` identifying the interface of the
        `modules` which will be monitored (e.g., the front-interface of all
        linear/convolutional layers), the list of `modules` to monitor, and a
        dictionary specifying the statistic to be tracked (`hook_tag`) and the
        functions needed to perform the tracking (`hook_setup`: this is a list
        composed of a protohook, the suitable call to the TB writer which will
        log the desired statistic, e.g., `add_scalar` or `add_histogram`, and
        possibly some keyword arguments to configure this call).
        """
        super(_HookTracker, self).__init__(writer)
        self.tag = tag
        self.hooks_generators = hooks_generators
        self.hooks_dict = {k: {n: None for n in modules} for k in self.hooks_generators.keys()}

    @staticmethod
    def get_type(hook):
        if hook.__name__ in __FORWARD_HOOKS__:
            type = 'forward'
        elif hook.__name__ in __BACKWARD_HOOKS__:
            type = 'backward'
        else:
            raise ValueError
        return type

    def setup(self, net, step):
        for hook_tag, hook_setup in self.hooks_generators.items():
            protohook = hook_setup[0]
            hook_type = self.get_type(protohook)
            kwargs = {} if len(hook_setup) == 2 else hook_setup[2]
            writer_gen = hook_setup[1](self.writer)  # this is a lambda expression (4 free variables)
            writer_gen_k = writer_gen(kwargs)  # this is still a lambda expression (3 free variables)
            writer_gen_k_e = writer_gen_k(step)  # this is still a lambda expression (2 free variables)
            for n, m in net.named_modules():
                if n in self.hooks_dict[hook_tag].keys():
                    writer_gen_k_e_t = writer_gen_k_e(self.tag+hook_tag+'/'+ntag)  # this is still a lambda expression (1 free variable)
                    hook_fn = partial(protohook, writer_gen_k_e_t)  # currying yields a function with signature (module, input, output)
                    if hook_type == 'forward':
                        handle = m.register_forward_hook(hook_fn)
                    elif hook_type == 'backward':
                        handle = m.register_backward_hook(hook_fn)
                    self.hooks_dict[hook_tag][n] = handle

    def release(self):
        for hook_tag, hooks in self.hooks_dict.items():
            for n, h in hooks.items():
                h.remove()
                self.hooks_dict[hook_tag][n] = None


class LinearOutputTracker(_HookTracker):
    def __init__(self, writer, modules):
        hooks_generators = {
            '_output_distribution': [protohook_fw_out_dist, tb_histogram_logger_generator],
            '_gradin_distribution': [protohook_bw_in_dist, tb_histogram_logger_generator]
        }
        super(LinearOutputTracker, self).__init__(writer, 'Linear', modules, hooks_generators)


class BatchNormInputTracker(_HookTracker):
    def __init__(self, writer, modules):
        hooks_generators = {
            '_input_per-channel_means_distribution': [protohook_fw_in_per_ch_mean_dist, tb_histogram_logger_generator],
            '_input_per-channel_stddevs_distribution': [protohook_fw_in_per_ch_std_dist, tb_histogram_logger_generator],
            # '_input_per-channel_stddevs_min': [protohook_fw_in_per_ch_std_min, tb_scalar_logger_generator],
            '_gradout_per-channel_norms_distribution': [protohook_bw_out_per_ch_norm_dist, tb_histogram_logger_generator]
        }
        super(BatchNormInputTracker, self).__init__(writer, 'Batch-normalisation', modules, hooks_generators)


class ActivationInputTracker(_HookTracker):
    def __init__(self, writer, modules):
        hooks_generators = {
            '_input_distribution': [protohook_fw_in_dist, tb_histogram_logger_generator],
            '_gradout_distribution': [protohook_bw_out_dist, tb_histogram_logger_generator]
        }
        super(ActivationInputTracker, self).__init__(writer, 'Activations', modules, hooks_generators)
