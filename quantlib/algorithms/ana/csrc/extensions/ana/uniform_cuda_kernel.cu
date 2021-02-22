#include <torch/extension.h>
#include <vector>

// #include <stdio.h>  // for debug

#include <cuda.h>
#include <cuda_runtime.h>


#define THREADS_PER_BLOCK 1024

#define ABS(x) ((x < 0.0f) ? -x : x)
#define CLAMP_0_1(x) ((x > 1.0f) ? 1.0f : ((x < 0.0f) ?  0.0f : x))


// definitions of CUDA kernels (executed on: GPU)

template <typename scalar_t>
__global__ void uniform_forward_cuda_kernel(
    scalar_t * const __restrict__ x_out,
    const scalar_t * __restrict__ x_in,
    const int64_t len_x,
    const scalar_t * __restrict__ q,
    const scalar_t * __restrict__ t,
    const int64_t len_t,
    const scalar_t * __restrict__ fmu,
    const scalar_t * __restrict__ fsigma,
    const scalar_t * __restrict__ training
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < len_x)
    {
        float sum = q[0];

        for (int it = 0; it < len_t; ++it)
        {
            // input position relative to the threshold
            float x_minus_t  = x_in[ix] - t[it] - *fmu;

            // expected value of the Heaviside function is the CDF of the uniform distribution
            float cdf;
            if (*training && (*fsigma != 0.0f))
            {
                float sigma_inv = 1.0f / (*fsigma);
                cdf = CLAMP_0_1((0.5f * x_minus_t) * sigma_inv + 0.5f);
            }
            else
            {
                cdf = (float) (x_minus_t >= 0.0f); // use the Heaviside which maps zero to one
            }

            // dilate and accumulate expected step value
            float dq = q[it + 1] - q[it];
            sum += dq * cdf;
        }

        x_out[ix] = sum;
    }
    else  // I am out of bounds!
    {
        return;
    }
}


template <typename scalar_t>
__global__ void uniform_backward_cuda_kernel(
    scalar_t * const __restrict__ grad_out,
    const scalar_t * __restrict__ grad_in,
    const scalar_t * __restrict__ x_in,
    const int64_t len_x,
    const scalar_t * __restrict__ q,
    const scalar_t * __restrict__ t,
    const int64_t len_t,
    const scalar_t * __restrict__ bmu,
    const scalar_t * __restrict__ bsigma
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < len_x)
    {
        float sum = 0.0f;

        for (int it = 0; it < len_t; ++it)
        {
            // input position relative to the threshold
            float x_minus_t = x_in[ix] - t[it] - *bmu;

            // the derivative of the expected (i.e., regularised) step function is the PDF of the uniform distribution
            float pdf;
            if (*bsigma != 0.0f)
            {
                float sigma_inv = 1.0f / (*bsigma);
                float local_derivative = (float) (ABS(x_minus_t) <= (*bsigma));
                pdf = 0.5f * sigma_inv * local_derivative;
            }
            else
            {
                pdf = 0.0f;  // no noise, no gradient!
            }

            // dilate and accumulate expected derivative
            float dq = q[it + 1] - q[it];
            sum += dq * pdf;
        }

        // compose gradients
        grad_out[ix] = sum * grad_in[ix];
    }
    else  // I am out of bounds!
    {
        return;
    }
}


// definitions of C++\CUDA interface (executed on: CPU)
// goals:
//   * allocate GPU memory for the output;
//   * define the parameters for the GPU kernel;
//   * call the kernel;

torch::Tensor uniform_forward_cuda_dispatch(
    torch::Tensor x_in,
    torch::Tensor q,
    torch::Tensor t,
    torch::Tensor fmu,
    torch::Tensor fsigma,
    torch::Tensor training
)
{
    auto x_out = torch::zeros_like(x_in);
    const dim3 blocks((x_in.numel() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES(
        x_in.type(),
        "uniform_forward_cuda",
        ([&] {
            uniform_forward_cuda_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
                x_out.data_ptr<scalar_t>(),
                x_in.data_ptr<scalar_t>(),
                x_in.numel(),
                q.data_ptr<scalar_t>(),
                t.data_ptr<scalar_t>(),
                t.numel(),
                fmu.data_ptr<scalar_t>(),
                fsigma.data_ptr<scalar_t>(),
                training.data_ptr<scalar_t>()
            );
        })
    );

    return x_out;
}


torch::Tensor uniform_backward_cuda_dispatch(
    torch::Tensor grad_in,
    torch::Tensor x_in,
    torch::Tensor q,
    torch::Tensor t,
    torch::Tensor bmu,
    torch::Tensor bsigma
)
{
    auto grad_out = torch::zeros_like(x_in);
    const dim3 blocks((x_in.numel() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES(
        x_in.type(),
        "uniform_backward_cuda",
        ([&] {
            uniform_backward_cuda_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
                grad_out.data_ptr<scalar_t>(),
                grad_in.data_ptr<scalar_t>(),
                x_in.data_ptr<scalar_t>(),
                x_in.numel(),
                q.data_ptr<scalar_t>(),
                t.data_ptr<scalar_t>(),
                t.numel(),
                bmu.data_ptr<scalar_t>(),
                bsigma.data_ptr<scalar_t>()
            );
        })
    );

    return grad_out;
}
