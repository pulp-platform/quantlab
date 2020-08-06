import os
import math
import numpy as np

from itertools import product
import torch
from functools import reduce
from operator import mul, add
import shutil


def export_tw(weight, weight_name, export_dir=os.path.curdir, T_out=16, T_in=16, align=16):
    """

    :param weight: the `torch.Tensor` that needs to be converted to binary
    :param T_out: the tiling size in the output channels dimension
    :param T_in: the tiling size in the input channels dimension
    :param align: the byte-alignment of the final binary file
    :return:
    """

    f2b = {
        -1.0: '11',
        0.0:  '00',
        1.0:  '01'
    }  # conversion dictionary from floating point to binary

    assert weight.ndim == 4  # must be a convolutional filter

    assert T_in % 4 == 0  # four ternary values are packed in one byte
    assert (T_out * (T_in // 4)) % align == 0  # each tile must be "alignable"

    C_out, C_in, K1, K2 = weight.shape  # K1 = filter height, K2 = filter width

    N2 = math.ceil(C_in / T_in)  # how many tiles do I need to cover a complete filter (i.e., all the input channels)?
    N1 = math.ceil(C_out / T_out)  # how many groups of tiles (each containing N2xK2xK1 tiles) do I need to cover the complete tensor?

    output = np.zeros((N1 * K1 * K2 * N2, T_out, T_in // 4), dtype=np.byte)

    for i1 in range(0, N1):
        for i_h in range(0, K1):
            for i_w in range(0, K2):
                for i2 in range(0, N2):

                    idx = (N2 * K2 * K1) * i1 + (N2 * K2) * i_h + (N2) * i_w + i2

                    tile_f = np.zeros((T_out, T_in))
                    s1 = i1 * T_out
                    e1 = s1 + (T_out if ((i1 < N1 - 1) or (weight.shape[0] % T_out == 0)) else weight.shape[0] - s1)
                    s2 = i2 * T_in
                    e2 = s2 + (T_in if ((i2 < N2 - 1) or (weight.shape[1] % T_in == 0)) else weight.shape[1] - s2)
                    tile_f[0:(e1 - s1), 0:(e2 - s2)] = weight[s1:e1, s2:e2, i_h, i_w]  # pad with 0s

                    tile_b = np.zeros((T_out, T_in // 4), dtype=np.byte)  # convert each sequence of 4 floats to a single byte

                    for j in range(0, T_out * T_in, 4):

                        r = j // T_in
                        c = j % T_in

                        slice = tile_f[r, c:(c + 4)]
                        bitstr = reduce(add, [f2b[slice[k].item()] for k in range(3, -1, -1)])
                        # bitstr = ''
                        # for k in range(3, -1, -1):
                        #     bitstr += f2b[slice[k].item()]

                        tile_b[r, c // 4] = np.array(int(bitstr, 2)).astype(np.byte)

                    output[idx, :] = tile_b

    with open(os.path.join(export_dir, weight_name), 'wb') as fp:
        fp.write(output.flatten())


def import_tw(weight, weight_name, export_dir=os.path.curdir, T_out=16, T_in=16, align=16):

    f2b = {
        -1.0: '11',
        0.0:  '00',
        1.0:  '01'
    }  # conversion dictionary from floating point to binary
    b2f = {v: k for k, v in f2b.items()}

    assert T_in % 4 == 0
    assert (T_out * (T_in // 4)) % 16 == 0

    C_out, C_in, K1, K2 = weight.shape

    N1 = math.ceil(C_out / T_out)
    N2 = math.ceil(C_in / T_in)

    with open(os.path.join(export_dir, weight_name), 'rb') as fp:
        buffer = np.frombuffer(fp.read(), dtype=np.byte)
        assert len(buffer) == reduce(mul, [N1 * K1 * K2 * N2, T_out, T_in // 4])
    buffer = buffer.reshape((N1 * K1 * K2 * N2, T_out, T_in // 4))

    for i1 in range(0, N1):
        for i_h in range(0, K1):
            for i_w in range(0, K2):
                for i2 in range(0, N2):

                    idx = (N2 * K2 * K1) * i1 + (N2 * K2) * i_h + (N2) * i_w + i2

                    tile_b = buffer[idx]
                    tile_f = np.zeros((T_out, T_in), dtype=np.float32)

                    for j in range(T_out * (T_in // 4)):

                        rb = j // (T_in // 4)
                        cb = j % (T_in // 4)

                        rf = (j * 4) // T_in
                        cf = (j * 4) % T_in

                        bitstring = np.binary_repr(tile_b[rb, cb], width=8)
                        for k in range(0, 4):
                            tile_f[rf, cf + 4 - k - 1] = b2f[bitstring[(2 * k):(2 * (k + 1))]]

                    s1 = i1 * T_out
                    e1 = s1 + (T_out if ((i1 < N1 - 1) or (weight.shape[0] % T_out == 0)) else weight.shape[0] - s1)
                    s2 = i2 * T_in
                    e2 = s2 + (T_in if ((i2 < N2 - 1) or (weight.shape[1] % T_in == 0)) else weight.shape[1] - s2)
                    weight[s1:e1, s2:e2, i_h, i_w] = torch.from_numpy(tile_f[0:(e1 - s1), 0:(e2 - s2)])

    return weight


if __name__ == '__main__':

    export_dir = 'tw_test'
    os.makedirs(export_dir, exist_ok=True)

    # test multiple combinations of C_out/C_in channels
    n_channels = [2**i for i in range(6, 11)]
    for C_out, C_in in product(n_channels, n_channels):

        filename = '{}x{}x3x3_tw.bin'.format(C_out, C_in)

        # export ternary tensor to binary file
        w_old = torch.randint(low=-1, high=2, size=(C_out, C_in, 3, 3), dtype=torch.float32)
        export_tw(w_old, filename, export_dir=export_dir)
        assert os.path.getsize(os.path.join(export_dir, filename)) == int(reduce(mul, w_old.shape) / 4)

        # import ternary tensor from binary file
        w_new = import_tw(torch.zeros_like(w_old, dtype=torch.float32), filename, export_dir=export_dir)
        assert torch.equal(w_new, w_old), "Imported tensor with shape {} differs from original tensor!".format(tuple(w_new.shape))

    shutil.rmtree(export_dir)
