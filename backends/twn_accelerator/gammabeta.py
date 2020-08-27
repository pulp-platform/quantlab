import os
import numpy as np
import shutil
import math


def export_gamma(gamma, gamma_name, export_dir=os.path.curdir, params, int_bits=10, frac_bits=17):
    """

    :param gamma: numpy array, float 64 (double)
    :param gamma_name:
    :param export_dir:
    """
    gamma_len = len(gamma)
    # zero pad gamma for export
    gamma_padded = np.zeros(params.n_blks(gamma_len) * params.blk_size)
    gamma_padded[0:gamma_len] = gamma
    quantum = 2**(-frac_bits)
    gamma_padded /= quantum
    gamma_padded = gamma_padded.astype('<u4')  # little endian unsigned integer 32

    max_n_quanta = 2**(int_bits + frac_bits) - 1
    gamma_temp = gamma_padded & np.array(max_n_quanta)
    assert np.all(gamma == gamma_temp), "{}/{}".format(np.sum(gamma != gamma_temp), len(gamma))  # each gamma is an UNSIGNED integer with maximum precision of 27 bits


    with open(os.path.join(export_dir, gamma_name), 'wb') as fp:
        fp.write(gamma_padded)


def import_gamma(gamma, gamma_name, params, export_dir=os.path.curdir):

    gamma_len = len(gamma)
    # pad gamma for checking imported data
    padded_len = params.n_blks(gamma_len) * params.blk_size
    with open(os.path.join(export_dir, gamma_name), 'rb') as fp:
        buffer = np.frombuffer(fp.read(), dtype='<u4')
        assert padded_len == len(buffer)

    # remove padding from imported data
    gamma = buffer.astype(np.float64)
    gamma = gamma[0:gamma_len]

    return gamma


def export_beta(beta, beta_name, params, export_dir=os.path.curdir, int_bits=8, frac_bits=17, true_frac_bits=17):
    # zero pad beta for export
    beta_len = len(beta)
    beta_padded = np.zeros(params.n_blks(beta_len)*params.blk_size)
    beta_padded[0:beta_len] = beta
    beta = beta_padded


    quantum = 2**(-frac_bits)
    beta /= quantum
    beta = beta.astype(np.int64)




    assert true_frac_bits <= frac_bits
    # how many bytes do I need to store this parameter?
    n_bytes = math.ceil((int_bits+true_frac_bits) / 2**3)  # ceil(n_bits / 8)
    n_bytes = 2**math.ceil(math.log(n_bytes, 2))  # allowed formats are 1, 2, 4, 8 bytes
    n_bytes = str(n_bytes)

    unused_bits = frac_bits - true_frac_bits

    probs = (beta % 2**unused_bits) / 2**unused_bits

    beta //= 2**unused_bits   # truncate fractional part

    # beta += 2 * np.random.binomial(1, 0.5, size=(len(beta),)) - 1  # probabilistic - random change
    # beta += np.random.binomial(1, probs)  # probabilistic - bias "correction"
    beta += (probs >= 0.5).astype(np.int64)  # deterministic - rounding (minimise error probability under assumtion of uniform distribution of inputs)
    # beta += (probs > 0.).astype(np.int64)  # deterministic - ceiling bias "correction"

    assert np.all(np.logical_and(np.abs(beta) <= (2**(int_bits+true_frac_bits) / 2), beta != (2**(int_bits+true_frac_bits) / 2)))  # each beta is a SIGNED integer with `int_bits` precision
    beta = beta.astype('i'+n_bytes)

    with open(os.path.join(export_dir, beta_name), 'wb') as fp:
        fp.write(beta)


def import_beta(beta, beta_name, params, export_dir=os.path.curdir, int_bits=8, frac_bits=17, true_frac_bits=17):

    assert true_frac_bits <= frac_bits
    beta_len = len(beta)
    padded_len = params.blk_size * params.n_blks(beta_len)


    # how many bytes did I need to store this parameter?
    n_bytes = math.ceil((int_bits+true_frac_bits) / 2**3)  # ceil(n_bits / 8)
    n_bytes = 2**math.ceil(math.log(n_bytes, 2))  # allowed formats are 1, 2, 4, 8 bytes
    n_bytes = str(n_bytes)

    with open(os.path.join(export_dir, beta_name), 'rb') as fp:
        buffer = np.frombuffer(fp.read(), dtype='i'+n_bytes)
        assert len(buffer) == padded_len

    unused_bits = frac_bits - true_frac_bits
    beta = buffer.astype(np.int64) * 2**unused_bits
    beta = beta.astype(np.float64)
    beta = beta[0:beta_len]

    return beta


if __name__ == '__main__':

    export_dir = 'gammabeta_test'
    os.makedirs(export_dir, exist_ok=True)

    gamma_bits = 27
    gamma_int_bits = 10
    frac_bits = gamma_bits - gamma_int_bits

    beta_int_bits = 8

    quantum = 2**(-frac_bits)

    # test multiple options for the number of channels
    n_channels = [2**i for i in range(6, 11)]
    for n in n_channels:

        filename_gamma = '{}_gamma.bin'.format(n)
        gamma_old = np.random.randint(low=0, high=2**(gamma_int_bits + frac_bits), size=(n,)).astype(np.float64) * quantum
        export_gamma(gamma_old, filename_gamma, export_dir=export_dir, int_bits=gamma_int_bits, frac_bits=frac_bits)
        assert os.path.getsize(os.path.join(export_dir, filename_gamma)) == 4 * len(gamma_old)

        gamma_new = np.zeros_like(gamma_old, dtype=np.float64)
        gamma_new = import_gamma(gamma_new, filename_gamma, export_dir=export_dir)
        assert np.all(gamma_new == gamma_old), "Imported gamma tensor with shape {} differs from original tensor!".format(tuple(gamma_new.shape))

        filename_beta = '{}_beta.bin'.format(n)
        beta_old = np.random.randint(low=-2**(beta_int_bits - 1), high=2**(beta_int_bits - 1), size=(n,)).astype(np.float64) * (2**frac_bits) * quantum
        export_beta(beta_old, filename_beta, export_dir=export_dir, int_bits=beta_int_bits, frac_bits=frac_bits)
        assert os.path.getsize(os.path.join(export_dir, filename_beta)) == len(beta_old)

        beta_new = np.zeros_like(beta_old, dtype=np.float64)
        beta_new = import_beta(beta_new, filename_beta, export_dir=export_dir)
        assert np.all(beta_new == beta_old)

    shutil.rmtree(export_dir)
