from setuptools import setup
from torch.utils import cpp_extension

setup(name='ana',
      ext_modules=[
          cpp_extension.CUDAExtension('ana_uniform_cuda', ['uniform_cuda.cpp', 'uniform_cuda_kernel.cu']),
          cpp_extension.CUDAExtension('ana_triangular_cuda', ['triangular_cuda.cpp', 'triangular_cuda_kernel.cu']),
          cpp_extension.CUDAExtension('ana_normal_cuda', ['normal_cuda.cpp', 'normal_cuda_kernel.cu']),
          cpp_extension.CUDAExtension('ana_logistic_cuda', ['logistic_cuda.cpp', 'logistic_cuda_kernel.cu']),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
