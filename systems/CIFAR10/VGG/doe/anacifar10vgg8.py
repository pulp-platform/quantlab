from enum import IntEnum
from typing import NamedTuple
import itertools
import math
from scipy.special import erfinv

from quantlib.algorithms.ana.ops import NoiseType, ForwardComputationStrategy
from manager.doeflows.experimentaldesign import ExperimentalSetup, ExperimentalDesign, patch_dictionary

from typing import List, Tuple


# ====================
#  DEGREES OF FREEDOM
# ====================

class Period(IntEnum):
    # SHORT    = 20
    STANDARD = 50
    # LONG     = 80


class NoiseMean(IntEnum):
    STATIC  = 0
    DYNAMIC = 1


class NoiseVariance(IntEnum):
    STATIC  = 0  # gradients flow in the same way throughout the training
    DYNAMIC = 1


class DecayIntervals(IntEnum):
    OVERLAPPED = 0
    SAME_START = 1
    SAME_END   = 2
    PARTITION  = 3


class DecayPowerLaw(IntEnum):
    HOMOGENEOUS = 0
    PROGRESSIVE = 1  # theory-inspired


NoisePolicy = NamedTuple('NoisePolicy', [('mi',    NoiseMean),
                                         ('sigma', NoiseVariance)])

DecayPolicy = NamedTuple('DecayPolicy', [('di',    DecayIntervals),
                                         ('dpl',   DecayPowerLaw)])

Policy      = NamedTuple('Policy',      [('fcs',   ForwardComputationStrategy),
                                         ('np',    NoisePolicy),
                                         ('dp',    DecayPolicy)])


def get_noise_hyperparameters(n_layers: int,
                              layer_id: int,
                              nt:       NoiseType,
                              np:       NoisePolicy,
                              dp:       DecayPolicy):

    # compute decay powers
    if dp.dpl == DecayPowerLaw.HOMOGENEOUS:
        base_power = 1
    elif dp.dpl == DecayPowerLaw.PROGRESSIVE:
        assert 0 <= layer_id < n_layers
        base_power = n_layers - layer_id
    else:
        raise ValueError

    mi_alpha    = 1 * base_power
    sigma_alpha = 2 * base_power

    # I compare different mass distributions in a given interval
    half_interval           = 0.5   # probability mass should be contained in the interval [mi_beta - half_interval, mi_beta + half_interval]
    mass_fraction_unbounded = 0.95  # probability distributions with unbounded support can not be "squeezed" inside the interval, so I try to put a "reasonable" amount of mass in it

    # compute mean hyper-parameters
    if np.mi == NoiseMean.STATIC:
        mi_beta  = 0.0
        mi_alpha = 0
    elif np.mi == NoiseMean.DYNAMIC:
        mi_beta = -half_interval  # anneal to the Heaviside H+
    else:
        raise ValueError

    # compute standard deviation (or standard deviation proxy) hyper-parameters
    if nt == NoiseType.UNIFORM:
        # for the uniform noise, I don't use the standard deviation, but a proxy s = \sqrt{3} * std
        sigma_beta = half_interval  # STE-like
    elif nt == NoiseType.TRIANGULAR:
        # for the triangular noise, I don't use the standard deviation, but a proxy s = \sqrt{6} * std
        sigma_beta = half_interval
    elif nt == NoiseType.LOGISTIC:
        sigma_beta = 1 / (2 * math.log((1 + mass_fraction_unbounded) / (1 - mass_fraction_unbounded)))
    elif nt == NoiseType.NORMAL:
        sigma_beta = - half_interval * math.sqrt(2) / (2 * erfinv(-mass_fraction_unbounded))
    else:
        raise ValueError
    assert sigma_beta > 0.0  # if there is no noise, gradients can't flow!

    if np.sigma == NoiseVariance.STATIC:
        sigma_alpha = 0
    elif np.sigma == NoiseVariance.DYNAMIC:
        pass
    else:
        raise ValueError

    return (mi_beta, mi_alpha), (sigma_beta, sigma_alpha)


def compute_noise_hyperparameters(n_layers: int,
                                  nt:       NoiseType,
                                  np:       NoisePolicy,
                                  dp:       DecayPolicy) -> List[Tuple[Tuple[float, int], Tuple[float, int]]]:
    return [get_noise_hyperparameters(n_layers, layer_id, nt, np, dp) for layer_id in range(0, n_layers)]


def compute_decay_intervals(n_layers: int,
                            period:   int,
                            dp:       DecayPolicy) -> List[Tuple[int, int]]:

    if dp.di == DecayIntervals.OVERLAPPED:
        starts = [period * 0        for layer_id in range(0, n_layers)]
        ends   = [period * n_layers for layer_id in range(0, n_layers)]

    elif dp.di == DecayIntervals.PARTITION:
        starts = [period * layer_id       for layer_id in range(0, n_layers)]
        ends   = [period * (layer_id + 1) for layer_id in range(0, n_layers)]

    elif dp.di == DecayIntervals.SAME_START:
        starts = [period * 0              for layer_id in range(0, n_layers)]
        ends   = [period * (layer_id + 1) for layer_id in range(0, n_layers)]

    elif dp.di == DecayIntervals.SAME_END:
        starts = [period * (n_layers - (layer_id + 1)) for layer_id in range(0, n_layers)]
        ends   = [period * n_layers                    for layer_id in range(0, n_layers)]

    else:
        raise ValueError

    decay_intervals = [(s, e) for s, e in zip(starts, ends)]

    return decay_intervals


def compute_timer_specs(layers: List[List[str]],
                        period: int,
                        nt:     NoiseType,
                        policy: Policy):

    noise_hyperparameters = compute_noise_hyperparameters(len(layers), nt, policy.np, policy.dp)
    decay_intervals       = compute_decay_intervals(len(layers), period, policy.dp)

    timer_specs = []
    for layer_id, (modules, ((mb, ma), (sb, sa)), (s, e)) in enumerate(zip(layers, noise_hyperparameters, decay_intervals)):
        timer_spec = dict()
        timer_spec['modules'] = modules
        timer_spec['mi'] = \
        {
            'beta': mb,
            'fun': 'bws',
            'kwargs': {
                   'tstart': s,
                   'tend':   e,
                   'alpha':  ma
            }
        }
        timer_spec['sigma'] = \
        {
            'beta': sb,
            'fun': 'bws',
            'kwargs': {
                'tstart': s,
                'tend':   e,
                'alpha':  sa
            }
        }
        timer_specs.append(timer_spec)

    return timer_specs


class ANACIFAR10VGG8(ExperimentalDesign):

    def __init__(self):
        dofs = (Period, NoiseType, ForwardComputationStrategy, NoiseMean, NoiseVariance, DecayIntervals, DecayPowerLaw)
        super(ANACIFAR10VGG8, self).__init__(dofs)

    def _generate_experimental_setups(self):

        setups = list()

        layers = [
            # conv/linear     activation
            ['pilot.0',      'pilot.2'],
            ['features.1',   'features.3'],
            ['features.4',   'features.6'],
            ['features.8',   'features.10'],
            ['features.11',  'features.13'],
            ['classifier.0', 'classifier.2'],
            ['classifier.3', 'classifier.5']
        ]

        n_layers     = len(layers)
        magic_number = 10 - n_layers

        noise_policies = {NoisePolicy(nm, nv)  for nm, nv      in itertools.product(NoiseMean, NoiseVariance)}
        decay_policies = {DecayPolicy(di, dpl) for di, dpl     in itertools.product(DecayIntervals, DecayPowerLaw)}
        policies       = {Policy(fcs, np, dp)  for fcs, np, dp in itertools.product(ForwardComputationStrategy, noise_policies, decay_policies)}

        dynamic_policies = set(filter(lambda p: not (p.np.mi == NoiseMean.STATIC and p.np.sigma == NoiseVariance.STATIC), policies))
        static_policies  = policies.difference(dynamic_policies)
        dynamic_policies = set(filter(lambda p: not (p.np.sigma == NoiseVariance.STATIC and p.fcs == ForwardComputationStrategy.EXPECTATION), dynamic_policies))
        static_policies  = set(filter(lambda p: p.fcs != ForwardComputationStrategy.EXPECTATION and p.dp.di == DecayIntervals.OVERLAPPED and p.dp.dpl == DecayPowerLaw.HOMOGENEOUS, static_policies))

        for period, nt in itertools.product(Period, NoiseType):

            n_epochs = period * (n_layers + magic_number)
            training_n_epochs_patch = \
                {
                    'training': {
                        'n_epochs': n_epochs
                    }
                }

            hand_scheduler_epoch = period * (n_layers + 1)  # lower the learning rate to fine-tune the parameters
            training_gd_patch = \
                {
                    'training': {
                        'gd': {
                            'lr_sched': {
                                'class': 'HandScheduler',
                                'kwargs': {'schedule': {hand_scheduler_epoch: 0.1}}
                            }
                        }
                    }
                }

            # generate setups and patches
            for policy in (static_policies | dynamic_policies):

                network_quantize_patch = \
                    {
                        'network': {
                            'quantize': {
                                'kwargs': {
                                    'noise_type': nt.name.lower(),
                                    'strategy':   policy.fcs.name.lower()
                                }
                            }
                        }
                    }

                timer_specs = compute_timer_specs(layers, period, nt, policy)
                training_quantize_patch = \
                    {
                        'training': {
                            'quantize': {
                                'kwargs': {'ctrl_spec': timer_specs}
                            }
                        }
                    }

                setup = (period.name, nt.name, policy.fcs.name, policy.np.mi.name, policy.np.sigma.name, policy.dp.di.name, policy.dp.dpl.name)
                patch = {**network_quantize_patch}
                patch = patch_dictionary(patch, training_n_epochs_patch)
                patch = patch_dictionary(patch, training_gd_patch)
                patch = patch_dictionary(patch, training_quantize_patch)

                setups.append(ExperimentalSetup(dofs_values=setup, config_patch=patch))

        return setups
