import torch
import torch.nn.functional as functional

from collections import namedtuple


OperandsFQ = namedtuple('OperandsFQ', ['x', 'w', 'm', 's', 'g', 'b', 'n_in', 'm_in', 'ex_in', 'n_out', 'm_out', 'ex_out'])
OperandsTQ = namedtuple('OperandsTQ', ['x', 'w', 'g', 'b'])


def get_input_range(d_in, d_out, k, s, p, i):

    assert 0 <= i < d_out, "Index out of range!"
    assert 1 + s * (d_out - 1) + (k - 1) == d_in + 2 * p

    start = 0 + i * s - p
    end = start + (k - 1)

    pad_left = max(0, 0 - start)
    pad_right = max(0, end - (d_in - 1))

    start = max(0, start)
    end = min(d_in - 1, end)

    return (start, end), (pad_left, pad_right)


def get_operands_fq(coords, x_in, layer, n_in=0, m_in=0., inq_layer=False):

    b, c_out, i, j = coords
    B, C_in, H_in, W_in = x_in.shape

    print(coords)

    nodes = list(layer.children())
    conv = [n for n in nodes if 'Conv' in n.__class__.__name__][0]
    bn   = [n for n in nodes if 'BatchNorm' in n.__class__.__name__][0]
    pool = [n for n in nodes if 'Pool' in n.__class__.__name__]
    pool = pool[0] if pool else None
    ste  = [n for n in nodes if 'STE' in n.__class__.__name__][0]

    K_in = conv.kernel_size
    S_in = conv.stride
    P_in = conv.padding
    H_med = (H_in + 2 * P_in[0] - K_in[0]) // S_in[0] + 1
    W_med = (W_in + 2 * P_in[1] - K_in[1]) // S_in[1] + 1
    C_med = conv.out_channels

    if pool:
        K_med = pool.kernel_size
        S_med = pool.stride
        P_med = pool.padding
        H_out = (H_med + 2 * P_med - K_med) // S_med + 1
        W_out = (W_med + 2 * P_med - K_med) // S_med + 1
        C_out = conv.out_channels

        (sHmed, eHmed), _ = get_input_range(H_med, H_out, K_med, S_med, P_med, i)
        (sWmed, eWmed), _ = get_input_range(W_med, W_out, K_med, S_med, P_med, j)

        import itertools
        patches_coords = list(itertools.product([b], [c_out], range(sHmed, eHmed+1), range(sWmed, eWmed+1)))

    else:
        patches_coords = [coords]

    x = []
    # print(patches_coords)
    for b, c_0, i, j in patches_coords:

        # print(b, c_0, i, j)

        (sHin, eHin), (pt, pb) = get_input_range(H_in, H_med, K_in[0], S_in[0], P_in[0], i)
        (sWin, eWin), (pl, pr) = get_input_range(W_in, W_med, K_in[1], S_in[1], P_in[1], j)

        # print(sHin, eHin, '-', sWin, eWin)

        x.append(functional.pad(x_in[b, ..., sHin:eHin+1, sWin:eWin+1], [pl, pr, pt, pb], mode='constant', value=0.))

    if inq_layer:
        w = conv.weight_frozen[c_out]
    else:
        w = conv.weight[c_out]
    m = bn.running_mean[c_out]
    s = torch.sqrt(bn.running_var[c_out] + bn.eps)
    g = bn.weight[c_out]
    b = bn.bias[c_out]

    ex_in = (2 * m_in) / (n_in - 1)
    n_out = ste.num_levels
    m_out = ste.abs_max_value
    ex_out = (2 * m_out) / (n_out - 1)

    operands = OperandsFQ(x, w, m, s, g, b, n_in, m_in, ex_in, n_out, m_out, ex_out)

    return operands


def get_operands_tq(coords, x_in, layer, d2d_layer=True):

    b, c_out, i, j = coords
    B, C_in, H_in, W_in = x_in.shape

    nodes = list(layer.children())
    conv_nodes = [n for n in nodes if 'Conv' in n.__class__.__name__]
    pool = [n for n in nodes if 'Pool' in n.__class__.__name__]
    pool = pool[0] if pool else None

    K_in = conv_nodes[0].kernel_size
    S_in = conv_nodes[0].stride
    P_in = conv_nodes[0].padding
    H_med = (H_in + 2 * P_in[0] - K_in[0]) // S_in[0] + 1
    W_med = (W_in + 2 * P_in[1] - K_in[1]) // S_in[1] + 1
    C_med = conv_nodes[0].out_channels

    if pool:
        K_med = pool.kernel_size
        S_med = pool.stride
        P_med = pool.padding
        H_out = (H_med + 2 * P_med - K_med) // S_med + 1
        W_out = (W_med + 2 * P_med - K_med) // S_med + 1
        C_out = conv_nodes[0].out_channels

        (sHmed, eHmed), _ = get_input_range(H_med, H_out, K_med, S_med, P_med, i)
        (sWmed, eWmed), _ = get_input_range(W_med, W_out, K_med, S_med, P_med, j)

        import itertools
        patches_coords = list(itertools.product([b], [c_out], range(sHmed, eHmed+1), range(sWmed, eWmed+1)))

    else:
        patches_coords = [coords]

    x = []
    for b, c_0, i, j in patches_coords:

        (sHin, eHin), (pt, pb) = get_input_range(H_in, H_med, K_in[0], S_in[0], P_in[0], i)
        (sWin, eWin), (pl, pr) = get_input_range(W_in, W_med, K_in[1], S_in[1], P_in[1], j)

        x.append(functional.pad(x_in[b, ..., sHin:eHin+1, sWin:eWin+1], [pl, pr, pt, pb], mode='constant', value=0.))

    if d2d_layer:
        w = conv_nodes[0].weight[c_out]
        g = conv_nodes[1].weight[c_out]
        b = conv_nodes[1].bias[c_out]
    else:
        w = conv_nodes[0].weight[c_out]
        g = conv_nodes[0].weight[c_out]
        b = conv_nodes[0].bias[c_out]

    operands = OperandsTQ(x, w, g, b)

    return operands


def mimic_fq_forward(operands, i):

    s = torch.sum(torch.mul(operands.x[i], operands.w))
    print('Integer equivalent: {}'.format(s / operands.ex_in))

    out = (((s - operands.m) / operands.s) * operands.g + operands.b) / operands.ex_out

    return out


def mimic_tq_forward(operands, i, frac_bits=17):

    s = torch.sum(torch.mul(operands.x[i], operands.w))
    print('Integer dot product: {}'.format(s))

    out = s * operands.g + operands.b

    return out / 2**frac_bits
