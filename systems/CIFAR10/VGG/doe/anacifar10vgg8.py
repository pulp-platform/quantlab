import numpy as np
import math
import matplotlib.pyplot as plt
from collections import namedtuple
import os
import json

from quantlib.algorithms.ana.ana_controller import ANATimer

from typing import List, Tuple, Dict
from typing import Union


# def exhaust_timer(timer: ANATimer, max_t: int) -> List[NoiseState]:
#
#     noise_states = []
#
#     for t in range(0, max_t):
#         timer.step(t)
#         noise_states.append(NoiseState(timer.fnoise_mi, timer.fnoise_sigma, timer.bnoise_mi, timer.bnoise_sigma))
#
#     return noise_states


# def generate_schedules(timers: Tuple[ANATimer], max_t: int) -> Tuple[List[NoiseState]]:
#     from functools import partial
#     return tuple(map(partial(exhaust_timer, max_t=max_t), timers))
#
#
# def get_subschedules(schedules: Tuple[List[NoiseState]], what: str) -> Dict[str, Tuple[List[float]]]:
#     assert what in _NOISE_PARAMETERS
#     return tuple([[getattr(ns, what) for ns in schedule] for schedule in schedules])
#
#
# def show_schedules(schedules: Tuple[List[NoiseState]], what: str) -> None:
#
#     assert what in _NOISE_PARAMETERS
#
#     print("Schedules for parameter {:s}".format(what))
#     print(" ".join(["Layer:     "] + ["{:6d}".format(l) for l in range(0, len(schedules))]))
#
#     for t, noise_states in enumerate(zip(*schedules)):
#         print(" ".join(["t = {:4d} - "] + ["{:6.3f}".format(getattr(ns, what)) for ns in noise_states]))
#
#
# def draw_subschedules(schedules: Tuple[List[NoiseState]], what: str, critical_instants: Tuple[int]) -> None:
#
#     subschedules = get_subschedules(schedules, what)
#     min_value = min(min(s) for s in subschedules)
#     max_value = max(max(s) for s in subschedules)
#
#     n_schedules = len(schedules)
#     x_range = len(schedules[0])
#
#     # pretty formatting
#     if min_value * max_value < 0.0:
#         delta = max_value - min_value
#     else:
#         if min_value >= 0.0:
#             delta = max_value
#         else:  # max_value <= 0.0
#             delta = -min_value
#     y_gap   = math.ceil(delta * (1.0 + 0.5))  # 50% extra space
#     y_range = n_schedules * y_gap
#
#     for schedule_id in range(0, n_schedules):
#         plt.axhline(y_gap * schedule_id, 0, x_range)
#     for t in critical_instants:
#         plt.axvline(t, min(0, min_value), min(0 + y_range, min_value + y_range))
#
#     for schedule_id in range(0, n_schedules):
#         plt.plot(np.array(range(0, x_range)), np.array(subschedules[schedule_id]) + y_gap * schedule_id, c='k')
#     plt.show()


CriticalInstants = namedtuple('CriticalInstants', ['critical_instants', 'n_epochs'])



from enum import Enum, IntEnum
from collections import namedtuple
import itertools


class NoiseType(Enum):
    UNIFORM    = 'uniform'
    TRIANGULAR = 'triangular'
    LOGISTIC   = 'logistic'
    NORMAL     = 'normal'


class NoiseMean(IntEnum):
    STATIC  = 0
    DYNAMIC = 1


class NoiseComputationForward(IntEnum):
    EXPECTATION    = 0
    ARGMAXSAMPLING = 1


class NoiseVariance(IntEnum):
    STATIC  = 0  # gradients flow in the same way throughout the training
    DYNAMIC = 1


NoisePolicy = namedtuple('NoisePolicy', ['mean', 'fcomputation', 'variance'])


class NoiseParameters(object):

    def __init__(self,
                 fnoise_mi_beta:     float,
                 fnoise_mi_alpha:    Union[None, int],
                 fnoise_sigma_beta:  float,
                 fnoise_sigma_alpha: Union[None, int],
                 bnoise_mi_beta:     float,
                 bnoise_mi_alpha:    Union[None, int],
                 bnoise_sigma_beta:  float,
                 bnoise_sigma_alpha: Union[None, int]):

        self.fnoise_mi_beta     = fnoise_mi_beta
        self.fnoise_mi_alpha    = fnoise_mi_alpha
        self.fnoise_sigma_beta  = fnoise_sigma_beta
        self.fnoise_sigma_alpha = fnoise_sigma_alpha
        self.bnoise_mi_beta     = bnoise_mi_beta
        self.bnoise_mi_alpha    = bnoise_mi_alpha
        self.bnoise_sigma_beta  = bnoise_sigma_beta
        self.bnoise_sigma_alpha = bnoise_sigma_alpha

    def __repr__(self):
        return "NoiseParameters(fnoise_mi_beta={}, fnoise_mi_alpha={}, fnoise_sigma_beta={}, fnoise_sigma_alpha={}, bnoise_mi_beta={}, bnoise_mi_alpha={}, bnoise_sigma_beta={}, bnoise_sigma_alpha={})".format(self.fnoise_mi_beta, self.fnoise_mi_alpha, self.fnoise_sigma_beta, self.fnoise_sigma_alpha, self.bnoise_mi_beta, self.bnoise_mi_alpha, self.bnoise_sigma_beta, self.bnoise_sigma_alpha)


def get_noise_parameters_template(noise_policy: NoisePolicy) -> NoiseParameters:

    if noise_policy.mean == NoiseMean.STATIC:
        fnoise_mi_beta  = 0.0
        fnoise_mi_alpha = 0
    elif noise_policy.mean == NoiseMean.DYNAMIC:
        fnoise_mi_beta  = -1.0  # anneal to the Heaviside H+
        fnoise_mi_alpha = None
    else:
        raise ValueError
    # the hyper-parameters regulating the mean must be consistent in the forward and backward passes
    bnoise_mi_beta  = fnoise_mi_beta
    bnoise_mi_alpha = fnoise_mi_alpha

    if noise_policy.variance == NoiseVariance.STATIC:
        bnoise_sigma_beta  = 1.0
        bnoise_sigma_alpha = 0
    elif noise_policy.variance == NoiseVariance.DYNAMIC:
        bnoise_sigma_beta  = 1.0
        bnoise_sigma_alpha = None
    else:
        raise ValueError

    if noise_policy.fcomputation == NoiseComputationForward.EXPECTATION:
        fnoise_sigma_beta  = bnoise_sigma_beta
        fnoise_sigma_alpha = bnoise_sigma_alpha
    elif noise_policy.fcomputation == NoiseComputationForward.ARGMAXSAMPLING:
        fnoise_sigma_beta  = 0.0
        fnoise_sigma_alpha = 0
    else:
        raise ValueError

    return NoiseParameters(fnoise_mi_beta, fnoise_mi_alpha, fnoise_sigma_beta, fnoise_sigma_alpha, bnoise_mi_beta, bnoise_mi_alpha, bnoise_sigma_beta, bnoise_sigma_alpha)


class DecayIntervals(IntEnum):
    OVERLAPPED = 0
    SAME_START = 1
    SAME_END   = 2
    PARTITION  = 3


class DecayPowerLaw(IntEnum):
    PROGRESSIVE = 0  # theory-inspired
    SAME        = 1


def compute_alphas(decay_power_law_policy: int, n_layers: Union[None, int] = None, layer_id: Union[None, int] = None) -> Tuple[int, int]:

    if decay_power_law_policy == DecayPowerLaw.PROGRESSIVE:
        assert 0 <= layer_id < n_layers
        base_power = n_layers - layer_id
    elif decay_power_law_policy == DecayPowerLaw.SAME:
        base_power = 1
    else:
        raise ValueError

    mi_alpha    = 1 * base_power
    sigma_alpha = 2 * base_power

    return mi_alpha, sigma_alpha


class Period(IntEnum):
    SHORT   = 20
    MIDDLE  = 50
    LONG    = 80


def compute_decay_intervals(decay_intervals_policy: int, n_layers: int, period: int) -> List[Tuple[int, int]]:

    assert decay_intervals_policy in DecayIntervals

    if decay_intervals_policy == DecayIntervals.OVERLAPPED:
        starts = [period * 0        for layer_id in range(0, n_layers)]
        ends   = [period * n_layers for layer_id in range(0, n_layers)]
    elif decay_intervals_policy == DecayIntervals.PARTITION:
        starts = [period * layer_id       for layer_id in range(0, n_layers)]
        ends   = [period * (layer_id + 1) for layer_id in range(0, n_layers)]
    elif decay_intervals_policy == DecayIntervals.SAME_START:
        starts = [period * 0              for layer_id in range(0, n_layers)]
        ends   = [period * (layer_id + 1) for layer_id in range(0, n_layers)]
    elif decay_intervals_policy == DecayIntervals.SAME_END:
        starts = [period * (n_layers - (layer_id + 1)) for layer_id in range(0, n_layers)]
        ends   = [period * n_layers                    for layer_id in range(0, n_layers)]
    else:
        raise ValueError

    decay_intervals = [(s, e) for s, e in zip(starts, ends)]

    return decay_intervals


def compute_timer_specs(noise_parameters: NoiseParameters,
                        layers: List[List[str]],
                        period: int,
                        decay_intervals_policy: int,
                        decay_power_law_policy: int):

    n_layers = len(layers)

    decay_intervals = compute_decay_intervals(decay_intervals_policy, n_layers, period)
    timer_specs = []
    for layer_id, (modules, (s, e)) in enumerate(zip(layers, decay_intervals)):
        import copy
        np = copy.deepcopy(noise_parameters)
        mi_alpha, sigma_alpha = compute_alphas(decay_power_law_policy, n_layers=n_layers, layer_id=layer_id)
        np.fnoise_mi_alpha    = mi_alpha    if np.fnoise_mi_alpha    is None else np.fnoise_mi_alpha # NoiseParameter is `object` instead of `namedtuple` since `namedtuple`s are immutable collections
        np.fnoise_sigma_alpha = sigma_alpha if np.fnoise_sigma_alpha is None else np.fnoise_sigma_alpha
        np.bnoise_mi_alpha    = mi_alpha    if np.bnoise_mi_alpha    is None else np.bnoise_mi_alpha
        np.bnoise_sigma_alpha = sigma_alpha if np.bnoise_sigma_alpha is None else np.bnoise_sigma_alpha
        timer_spec = dict()
        timer_spec['modules'] = modules
        timer_spec['fnoise'] = {
            'mi': {
                'base': np.fnoise_mi_beta,
                'fun': 'lws',
                'kwargs': {
                       'tstart': s,
                       'tend':   e,
                       'alpha':  np.fnoise_mi_alpha}
                   },
            'sigma': {
                'base': np.fnoise_sigma_beta,
                'fun': 'lws',
                'kwargs': {
                    'tstart': s,
                    'tend':   e,
                    'alpha':  np.fnoise_sigma_alpha
                }
            }
        }
        timer_spec['bnoise'] = {
            'mi': {
                'base': np.bnoise_mi_beta,
                'fun': 'lws',
                'kwargs': {
                       'tstart': s,
                       'tend':   e,
                       'alpha':  np.bnoise_mi_alpha}
                   },
            'sigma': {
                'base': np.bnoise_sigma_beta,
                'fun': 'lws',
                'kwargs': {
                    'tstart': s,
                    'tend':   e,
                    'alpha':  np.bnoise_sigma_alpha
                }
            }
        }
        timer_specs.append(timer_spec)

    return timer_specs



from manager.doeflows.experimentaldesign import Configuration, ExperimentalDesign


class ANACIFAR10VGG8(ExperimentalDesign):

    def __init__(self):
        dofs = (Period, NoiseType, NoiseMean, NoiseVariance, NoiseComputationForward, DecayIntervals, DecayPowerLaw)
        super(ANACIFAR10VGG8, self).__init__(dofs)

    def _load_base_cfg(self):
        """Generate a basic configuration dictionary that will be patched."""
        with open(os.path.join(os.path.dirname(__file__), '.'.join([self.__class__.__name__, 'json'])), 'r') as fp:
            self._base_cfg = json.load(fp)

    def _generate_configs(self):

        configs = []

        layers = [
            # conv/linear     activation
            ['pilot.0', 'pilot.1'],
            ['features.1', 'features.3'],
            ['features.4', 'features.6'],
            ['features.8', 'features.10'],
            ['features.11', 'features.13'],
            ['classifier.0', 'classifier.2'],
            ['classifier.3', 'classifier.5']
        ]
        n_layers = len(layers)

        noise_policies = {NoisePolicy(nm, fc, nv) for nm, fc, nv in
                          itertools.product(NoiseMean, NoiseComputationForward, NoiseVariance)}
        noise_policies = set(filter(lambda np: not (
                    np.fcomputation == NoiseComputationForward.EXPECTATION and np.variance == NoiseVariance.STATIC),
                                    noise_policies))
        static_noise_policy = {np for np in noise_policies if (
                    np.mean == NoiseMean.STATIC and np.fcomputation == NoiseComputationForward.ARGMAXSAMPLING and np.variance == NoiseVariance.STATIC)}
        dynamic_noise_policies = noise_policies.difference(static_noise_policy)

        for nt in NoiseType:

            network_quantize_patch = {'network': {
                'quantize': {'noise_type': nt.value}}}  # dictionaries can be "patched" using the `update` method

            for p in Period:

                magic_number = 10 - n_layers
                n_epochs = p * (n_layers + magic_number)
                hand_scheduler_epoch = p * (n_layers + 1)  # lower the learning rate to fine-tune the parameters

                training_n_epochs_patch = {'training': {'n_epochs': n_epochs}}
                training_gd_lr_sched_patch = {'training': {'gd': {
                    'lr_sched': {'class': 'HandScheduler', 'kwargs': {'schedule': {hand_scheduler_epoch: 0.1}}}}}}

                # static noise policy
                npt = get_noise_parameters_template(next(iter(static_noise_policy)))

                timer_specs = compute_timer_specs(npt, layers, p, DecayIntervals.OVERLAPPED, DecayPowerLaw.SAME)
                training_quantize_patch = {'training': {'quantize': {'kwargs': {'ctrl_spec': timer_specs}}}}

                setup = (
                p.value, nt.value, *[a.value for a in next(iter(static_noise_policy))], DecayIntervals.OVERLAPPED.value,
                DecayPowerLaw.SAME.value)
                patch = {**network_quantize_patch, **training_n_epochs_patch, **training_gd_lr_sched_patch,
                         **training_quantize_patch}

                configs.append(Configuration(setup=setup, patch=patch))

                # dynamic noise policies
                for np in dynamic_noise_policies:

                    npt = get_noise_parameters_template(np)

                    for dip, dplp in itertools.product(DecayIntervals, DecayPowerLaw):
                        timer_specs = compute_timer_specs(npt, layers, p, dip, dplp)
                        training_quantize_patch = {'training': {'quantize': {'kwargs': {'ctrl_spec': timer_specs}}}}

                        setup = (p.value, nt.value, *[a.value for a in np], dip.value, dplp.value)
                        patch = {**network_quantize_patch, **training_n_epochs_patch, **training_gd_lr_sched_patch,
                                 **training_quantize_patch}
                        configs.append(Configuration(setup=setup, patch=patch))

        self._configs = configs
