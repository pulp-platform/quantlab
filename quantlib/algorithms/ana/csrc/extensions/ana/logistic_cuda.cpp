#include <torch/extension.h>
#include <vector>

// #include <stdio.h>  // for debug


// declarations of C++\CUDA interface (executed on: CPU)

torch::Tensor logistic_forward_cuda_dispatch(
    torch::Tensor x_in,
    torch::Tensor q,
    torch::Tensor t,
    torch::Tensor fmu,
    torch::Tensor fsigma,
    torch::Tensor training
);

torch::Tensor logistic_backward_cuda_dispatch(
    torch::Tensor grad_in,
    torch::Tensor x_in,
    torch::Tensor q,
    torch::Tensor t,
    torch::Tensor bmu,
    torch::Tensor bsigma
);


// definitions of C++ wrappers (executed on: CPU)
// goals:
//   * check that the memory layout of tensors allocated on GPU memory is correct
//   * call the dispatcher

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor logistic_forward_cuda(
    torch::Tensor x_in,
    torch::Tensor q,
    torch::Tensor t,
    torch::Tensor fmu,
    torch::Tensor fsigma,
    torch::Tensor training
)
{
    CHECK_INPUT(x_in);
    CHECK_INPUT(q);
    CHECK_INPUT(t);
    CHECK_INPUT(fmu);
    CHECK_INPUT(fsigma);
    CHECK_INPUT(training);

    return logistic_forward_cuda_dispatch(x_in, q, t, fmu, fsigma, training);
}


torch::Tensor logistic_backward_cuda(
    torch::Tensor grad_in,
    torch::Tensor x_in,
    torch::Tensor q,
    torch::Tensor t,
    torch::Tensor bmu,
    torch::Tensor bsigma
)
{
    CHECK_INPUT(grad_in);
    CHECK_INPUT(x_in);
    CHECK_INPUT(q);
    CHECK_INPUT(t);
    CHECK_INPUT(bmu);
    CHECK_INPUT(bsigma);

    return logistic_backward_cuda_dispatch(grad_in, x_in, q, t, bmu, bsigma);
}


// compile into a Python module

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &logistic_forward_cuda, "ANA logistic noise forward (CUDA)");
    m.def("backward", &logistic_backward_cuda, "ANA logistic noise backward (CUDA)");
}
