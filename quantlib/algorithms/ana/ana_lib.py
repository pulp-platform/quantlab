import torch
from . import ana_uniform
import ana_uniform_cuda
from . import ana_triangular
import ana_triangular_cuda
from . import ana_normal
import ana_normal_cuda
from . import ana_logistic
import ana_logistic_cuda


all = [
    'ANAUniform',
    'ANATriangular',
    'ANANormal',
    'ANALogistic',
]


# uniform noise

class ANAUniform(torch.autograd.Function):
    """A stochastic process composed by step functions.

    This class defines a stochastic process whose elementary events are step
    functions with fixed quantization levels (codominion) and uniform noise on
    the jumps positions.
    """
    @staticmethod
    def forward(ctx, x_in, q, t, fmu, fsigma, bmu, bsigma, training):
        ctx.save_for_backward(x_in, q, t, bmu, bsigma)
        if fsigma.is_cuda:
            x_out = ana_uniform_cuda.forward(x_in, q, t, fmu, fsigma, torch.Tensor([training]).to(fsigma))
        else:
            x_out = ana_uniform.forward(x_in, q, t, fmu, fsigma, training)
        return x_out

    @staticmethod
    def backward(ctx, grad_in):
        x_in, q, t, bmu, bsigma = ctx.saved_tensors
        if bsigma.is_cuda:
            grad_out = ana_uniform_cuda.backward(grad_in, x_in, q, t, bmu, bsigma)
        else:
            grad_out = ana_uniform.backward(grad_in, x_in, q, t, bmu, bsigma)
        return grad_out, None, None, None, None, None, None, None


# triangular noise

class ANATriangular(torch.autograd.Function):
    """A stochastic process composed by step functions.

    This class defines a stochastic process whose elementary events are step
    functions with fixed quantization levels (codominion) and triangular noise
    on the jumps positions.
    """
    @staticmethod
    def forward(ctx, x_in, q, t, fmu, fsigma, bmu, bsigma, training):
        ctx.save_for_backward(x_in, q, t, bmu, bsigma)
        if fsigma.is_cuda:
            x_out = ana_triangular_cuda.forward(x_in, q, t, fmu, fsigma, torch.Tensor([training]).to(fsigma))
        else:
            x_out = ana_triangular.forward(x_in, q, t, fmu, fsigma, training)
        return x_out

    @staticmethod
    def backward(ctx, grad_in):
        x_in, q, t, bmu, bsigma = ctx.saved_tensors
        if bsigma.is_cuda:
            grad_out = ana_triangular_cuda.backward(grad_in, x_in, q, t, bmu, bsigma)
        else:
            grad_out = ana_triangular.backward(grad_in, x_in, q, t, bmu, bsigma)
        return grad_out, None, None, None, None, None, None, None


# normal noise

class ANANormal(torch.autograd.Function):
    """A stochastic process composed by step functions.

    This class defines a stochastic process whose elementary events are step
    functions with fixed quantization levels (codominion) and normal noise on
    the jumps positions.
    """
    @staticmethod
    def forward(ctx, x_in, q, t, fmu, fsigma, bmu, bsigma, training):
        ctx.save_for_backward(x_in, q, t, bmu, bsigma)
        if fsigma.is_cuda:
            x_out = ana_normal_cuda.forward(x_in, q, t, fmu, fsigma, torch.Tensor([training]).to(fsigma))
        else:
            x_out = ana_normal.forward(x_in, q, t, fmu, fsigma, training)
        return x_out

    @staticmethod
    def backward(ctx, grad_in):
        x_in, q, t, bmu, bsigma = ctx.saved_tensors
        if bsigma.is_cuda:
            grad_out = ana_normal_cuda.backward(grad_in, x_in, q, t, bmu, bsigma)
        else:
            grad_out = ana_normal.backward(grad_in, x_in, q, t, bmu, bsigma)
        return grad_out, None, None, None, None, None, None, None


# logistic noise

class ANALogistic(torch.autograd.Function):
    """A stochastic process composed by step functions.

    This class defines a stochastic process whose elementary events are step
    functions with fixed quantization levels (codominion) and logistic noise on
    the jumps positions.
    """
    @staticmethod
    def forward(ctx, x_in, q, t, fmu, fsigma, bmu, bsigma, training):
        ctx.save_for_backward(x_in, q, t, bmu, bsigma)
        if fsigma.is_cuda:
            x_out = ana_logistic_cuda.forward(x_in, q, t, fmu, fsigma, torch.Tensor([training]).to(fsigma))
        else:
            x_out = ana_logistic.forward(x_in, q, t, fmu, fsigma, training)
        return x_out

    @staticmethod
    def backward(ctx, grad_in):
        x_in, q, t, bmu, bsigma = ctx.saved_tensors
        if bsigma.is_cuda:
            grad_out = ana_logistic_cuda.backward(grad_in, x_in, q, t, bmu, bsigma)
        else:
            grad_out = ana_logistic.backward(grad_in, x_in, q, t, bmu, bsigma)
        return grad_out, None, None, None, None, None, None, None
