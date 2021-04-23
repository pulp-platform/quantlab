import numpy as np
import torch


def cast_gamma(gamma_fp64, int_bits=10, frac_bits=17):

    quantum = 2**(-frac_bits)
    gamma_fp64 /= quantum
    gamma_uint32 = gamma_fp64.astype('<u4')  # little endian unsigned integer 32

    max_n_quanta = 2**(int_bits + frac_bits) - 1
    gamma_uint32_masked = gamma_uint32 & np.array(max_n_quanta)   # each gamma is an UNSIGNED integer with maximum precision of `int_bits + frac_bits` bits
    assert np.all(gamma_uint32 == gamma_uint32_masked)

    gamma_uint32_into_fp64 = gamma_uint32_masked.astype(np.float64)
    return gamma_uint32_into_fp64


def cast_beta(beta_fp64, int_bits=8, gamma_frac_bits=17, beta_frac_bits=0):

    assert beta_frac_bits <= gamma_frac_bits

    quantum = 2**(-gamma_frac_bits)
    beta_fp64 /= quantum
    beta_int32 = beta_fp64.astype(np.int32)

    n_truncated_bits = gamma_frac_bits - beta_frac_bits
    beta_int32 //= 2**n_truncated_bits   # truncate the fractional digits that can not be represented

    # probs = (beta_int32 % 2**n_truncated_bits) / 2**n_truncated_bits

    # # stochastic corrections (choose one)
    # # S1.
    # beta_int32 += 2 * np.random.binomial(1, 0.5, size=(len(beta_int32),)) - 1  # random +/-1
    # # S2.
    # beta_int32 += np.random.binomial(1, probs)  # round to the closest value with probability proportional to closeness

    # # deterministic corrections (choose one)
    # # D1.
    # beta_int32 += (probs >= 0.5).astype(np.int32)  # round to the closest value (should minimise the proability of errors under the assumption that the inputs are uniformly distributed)
    # beta_int32 += (probs > 0.).astype(np.int32)  # round to ceiling

    assert np.all(np.logical_and(np.abs(beta_int32) <= (2**(int_bits + beta_frac_bits) / 2), beta_int32 != (2**(int_bits + beta_frac_bits) / 2)))  # each beta is a SIGNED integer with `int_bits + beta_frac_bits` precision

    beta_int32 *= 2**n_truncated_bits
    beta_int32_into_fp64 = beta_int32.astype(np.float64)

    return beta_int32_into_fp64


def foldsteinqconvbnste(n_in, m_in, weight, mu, sigma, eps, gamma, beta, n_out, m_out, gamma_int_bits=10, gamma_frac_bits=17, beta_int_bits=8, beta_frac_bits=0):

    def torch2numpyfp64(x):
        return x.detach().cpu().numpy().astype(np.float64)

    m_in   = torch2numpyfp64(m_in)
    weight = torch2numpyfp64(weight)
    mu     = torch2numpyfp64(mu)
    sigma  = torch2numpyfp64(sigma)
    gamma  = torch2numpyfp64(gamma)
    beta   = torch2numpyfp64(beta)
    m_out  = torch2numpyfp64(m_out)

    # see batch normalisation docs in PyTorch
    sigma = np.sqrt(sigma + eps)
    # STE quanta
    eps_in  = (2 * m_in) / (n_in - 1)
    eps_out = (2 * m_out) / (n_out - 1)

    # REORDER

    # compensate for negative gammas
    flip    = np.sign(gamma)
    w_tmp   = weight.transpose(1, 2, 3, 0)
    w_tmp   *= flip
    fweight = w_tmp.transpose(3, 0, 1, 2)

    # fold gamma
    fgamma = (eps_in * gamma) / (eps_out * sigma)
    fgamma *= flip
    fgamma = fgamma.reshape(-1, 1, 1, 1)

    # fold beta
    # # reordering rule when `x` is replaced by `x + C`
    # fwsum = w.reshape((w.shape[0], -1)).sum(axis=1)
    # fbeta = ((((C * fwsum) - mu) * gamma / sigma) + beta) / eps_out  # "rounding" version
    # fbeta = ((((C * fwsum) - mu) * gamma / sigma) + beta) / eps_out + 0.5  # "flooring" version
    # standard reordering rule
    # fbeta = (((-mu * gamma) / sigma) + beta) / eps_out  # "rounding version
    fbeta = (((-mu * gamma) / sigma) + beta) / eps_out + 0.5  # "flooring" version

    # CAST

    fgamma = cast_gamma(fgamma, int_bits=gamma_int_bits, frac_bits=gamma_frac_bits)
    fbeta  = cast_beta(fbeta, int_bits=beta_int_bits, gamma_frac_bits=gamma_frac_bits, beta_frac_bits=beta_frac_bits)

    def numpy2torchfp64(x):
        return torch.from_numpy(x.astype(np.float64))

    return numpy2torchfp64(fweight), numpy2torchfp64(fgamma), numpy2torchfp64(fbeta)


def foldconvbnste(weight, mu, sigma, eps, gamma, beta, n_out, m_out):

    def torch2numpyfp64(x):
        return x.detach().cpu().numpy().astype(np.float64)

    weight = torch2numpyfp64(weight)
    mu     = torch2numpyfp64(mu)
    sigma  = torch2numpyfp64(sigma)
    gamma  = torch2numpyfp64(gamma)
    beta   = torch2numpyfp64(beta)
    m_out  = torch2numpyfp64(m_out)

    # see batch normalisation docs in PyTorch
    sigma = np.sqrt(sigma + eps)
    # STE quantum
    eps_out = (2 * m_out) / (n_out - 1)

    # REORDER

    fweight = (weight * (gamma / sigma).reshape(-1, 1, 1, 1)) / eps_out
    fbias   = ((-mu * gamma) / sigma + beta) / eps_out + 0.5

    def numpy2torchfp64(x):
        return torch.from_numpy(x.astype(np.float64))

    return numpy2torchfp64(fweight), numpy2torchfp64(fbias)


def foldsteinqconvbn(n_in, m_in, weight, mu, sigma, eps, gamma, beta):

    def torch2numpyfp64(x):
        return x.detach().cpu().numpy().astype(np.float64)

    m_in   = torch2numpyfp64(m_in)
    weight = torch2numpyfp64(weight)
    mu     = torch2numpyfp64(mu)
    sigma  = torch2numpyfp64(sigma)
    gamma  = torch2numpyfp64(gamma)
    beta   = torch2numpyfp64(beta)

    # see batch normalisation docs in PyTorch
    sigma = np.sqrt(sigma + eps)
    # STE quantum
    eps_in = (2 * m_in) / (n_in - 1)

    # REORDER

    fweight = weight * ((eps_in * gamma) / sigma).reshape(-1, 1, 1, 1)
    fbias   = (-mu * gamma) / sigma + beta + 0.5

    def numpy2torchfp64(x):
        return torch.from_numpy(x.astype(np.float64))

    return numpy2torchfp64(fweight), numpy2torchfp64(fbias)
