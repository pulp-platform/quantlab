Additive noise annealing (ANA)
==============================

.. automodule:: quantlib.algorithms.ana.ana_ops
   :members:


Writing C/C++ and CUDA extensions for PyTorch
---------------------------------------------

We used PyTorch's binding mechanics to expose our own CUDA kernels and accelerate the execution of ANA operations.
In this section, we will give you an overview of this process, navigating the call stack using ANA as an example.

If you are a PyTorch user, you are probably aware that the operations acting on :py:class:`~torch.Tensor` objects can be recorded by the autograd mechanics.
When this is the case, it is possible to compute gradients using reverse-mode automatic differentiation (i.e., back-propagation) and training deep neural networks that use the operation.
When you implement your custom module, apart from bookkeeping data structures and algorithm, you can encode this core functionality using objects of class :py:class:`torch.autograd.Function`.
A :py:class:`torch.autograd.Function` object can be registered when you instantiate a custom :py:class:`~torch.Module`.

For example, consider the code of the class :py:class:`ANAActivation` (defined in ``quantlib/algorithms/ana/ana_ops.py:L102``).
You can see that its ``forward`` method boils down to a call to ``self.ana_op`` (``L111``).
Since this attribute is not defined in the constructor method of the class, it must be inherited from parent classes.
Notice that the only parent class of ``ANAActivation`` is :py:class:`ANAModule`.
Therefore,  when calling the constructor method of the parent class (``L105``) there is no ambiguity to be solved resorting to Python's method resolution order (MRO).
The constructor method of ``ANAModule`` (``L47``) carries out two steps:

* defining the quantizer function (``L49``);
* preparing the noisy function that characterises the algorithm (``L50``).

In particular, following the execution of the ``setup_noise`` method (defined at line ``78``), we see that it imports a class from the ``ana_lib`` module, and that this class has an ``apply`` attribute.
If we open the file ``quantlib/algorithms/ana/ana_lib.py`` we see that it defines four sub-classes of :py:class:`torch.autograd.Function`.
As expected from the logic of automatic differentiation, these classes implement ``forward`` and ``backward`` methods.
The logic of these functions is the same, the only difference being the noise distribution that should be used when training ANA networks.
In particular, we see that there is a branching structure conditioned on two observations:

* the accelerated functions have been compiled and installed;
* the ``Tensor``s are stored on GPU (the implicit assumption in our implementation is that if one argument ``Tensor`` is on GPU, all of them should be).

Since the purpose of this section is illustrating how PyTorch CUDA extensions work, we assume that the accelerated functions are available.
To continue our traversal of the call stack, we focus on the :py:class:`ANAUniform` sub-class (which is defined in ``quantlab/algorithms/ana/ana_lib.py:L49``).
Consider its ``forward`` method.
Since we are interested in the call stack involving custom extensions, we imagine taking the GPU branch (``L60``).
This branch calls the function ``forward`` defined in the Python module named ``ana_uniform_cuda``.
This module is imported in the "header" of the ``ana_lib.py`` using an *absolute* kind of import.
This implementation choice points to the fact that the module is supposed to be installed in the global search path of the Python interpreter.
Indeed, this module is installed by applying Python's ``setuptools`` package through the script ``quantlib/algorithms/ana/csrc/setup.py``.
If you open this file, you will see that the ``ana_uniform_cuda`` module is defined as an *extension module* at line ``27``.
We will not delve into the details of ``setuptools`` here; if you are interested, we point you to the official documentation of the package. [...]
To follow our traversal of the call stack, all that you need to know is that installing ``ana_uniform_cuda`` by interpreting the ``setup.py`` script will install the module after building it from the source files defined in the ``sources`` list associated with the :py:class:`torch.utils.cpp_extension.CUDAExtension` object.
It is here that the "boundary" between Python/PyTorch and C/C++/CUDA is located.

Before we proceed, it is time to recall some concepts about C and C++ that are required to understand the completion of the stack traversal: *templatisation* and *Python bindings*.

The default implementation of the Python programming language is called *Cpython*.
As the name suggests, Cpython is written in C.
It is very likely that the Python interpreter and its basic constructs running on your machine have been compiled from C source files.
Such a Python interpreter manages C abstractions of type ``PyObject``.
[``pybind11``](https://pybind11.readthedocs.io/en/latest/) is a header-only library that facilitates the creation of ``PyObject``s wrapping existing C++ code.
As the name of the library suggest, it is designed to be compliant with the C++11 standard of the C++ language.

C and C++ are typed languages, which means that each function must specify which data types it will accept as inputs and which data type it will return as output.
This information is required by the compiler to disambiguate and direct the generation of assembly and machine code in a way that is compatible with the underlying hardware's ISA.
In C, those cases where the users need to implement the same function but operating on different data types (e.g., matrix-matrix products for matrices with integer components and matrix-matrix products for matrices with floating-point components), they would need to replicate the function's code and change the data types.
This required replication of functionally equivalent code (*boilerplate code*) can make maintenance time consuming and error-prone, since a function will probably need to be (manually) changed in the same way for multiple data types.
To circumvent this issue, C++ introduced *templatisation*.
Templatisation is a feature of C++ that allows users to write functions in a data-type-agnostic way.
The compiler will generate and compile code for each specified data type; we will see later how the choice for the supported data types can be made.

We can now go back to our traversal of the call stack.
Remember that in Python everything is an object.
Also Python modules (collections of names, i.e., namespaces that are the equivalent of libraries in C/C++) are objects.
The C/C++/CUDA files listed in the ``sources`` arguments to extension objects contain the information required to build the code that users will be able to import as Python modules.
If we consider the ``quantlib/algorithms/ana/csrc/uniform_cuda.cpp`` file, the ``PyObject`` associated to the ``ana_uniform_cuda`` module is created in the lines going from ``L100`` to ``L104``.
These lines make use of the C++ macro ``PYBIND11_MODULE`` imported from the ``torch/extension.h`` header file included at ``L22`` (since this is a ``pybind11`` macro, it is likely that it is included into the namespace defined by ``torch/extension.h`` by including ``pybind11`` header files in there, although I have not opened ``torch/extension.h`` to verify this).
The first argument to the macro, ``TORCH_EXTENSION_NAME``, is resolved to the value of the ``name`` keyword argument passed to the extension object in the ``setup.py`` file; in our example traversal, this is ``ana_uniform_cuda``.
The second argument to the macro will be a ``PyObject`` implementing a Python module.
The body of the macro (lines ``102`` and ``103``) adds two symbols to the namespace, ``forward`` and ``backward``.
These symbols are bound to pointers to the code of the actual functions that will be called: ``uniform_forward_cuda`` and ``uniform_backward_cuda``, respectively.
In this way, when the method ``ana_uniform_cuda.forward`` is invoked (as from ``quantlib/algorithms/ana/ana_lib.py:L60``), the code that will actually be executed is that described by the C++ function ``uniform_forward_cuda`` (implemented at ``quantlib/algorithms/ana/csrc/uniform_cuda.cpp:L58``).

Since we chose to traverse the call stack for the ``forward`` method only, we will now look into the final call stack induced by ``uniform_backward_cuda``.
First of all, this function checks that the ``torch.Tensor`` arguments are actually suitable arguments for the operation that it will perform.
Note that these checks are type-agnostic: i.e., they take in input ``torch.Tensors``, but do not depend on whether their "payload" (i.e., the arrays of numerical values that they wrap) is of a specific data type.
Indeed, they just check that the payload is located on GPU (``CHECK_CUDA``) and that their layout is consistent with the "dimension priority" established by their ``dims`` attribute (``CHECK_CONTIGUOUS``).
Loosely speaking, this last check is intended to avoid operating on *views*, i.e., ``torch.Tensor``s where changing by one unit the index associated with the last dimension does not return a pointer to an element stored at a memory location that is contiguous to the one where the previously-pointed-to element is stored.
After performing these checks, ``uniform_forward_cuda`` calls the ``uniform_forward_cuda_dispatch`` function.
This function is defined at ``quantlib/algorithms/ana/csrc/uniform_cuda_kernel.cu:L145``.
Its purpose is acting as a "switch" between multiple builds of the real CUDA kernel, ``uniform_forward_cuda_kernel`` (defined at ``quantlib/algorithms/ana/csrc/uniform_cuda_kernel.cu:L39``).
In fact, note that ``uniform_forward_cuda_kernel`` is a templetised function, depending on an abstract scalar data type ``scalar_t``.
``uniform_forward_cuda_dispatch`` uses the macro ``AT_DISPATCH_FLOATING_TYPES`` to expand a switch construct based on the data type of the payload.
Indeed, notice that ``x_in.type()`` is passed as an argument to the macro, and that the abstract scalar data type ``scalar_t`` is never referred outside the scope of the macro.
Loosely speaking, we might say that the ``uniform_forward_cuda_dispatch`` "gently queries" ``torch.Tensor`` objects for their data type, then selects the correct function to process their payload.

To recap, we will now perform a fast traversal of the call stack, listing the files where the symbol is located:

* ``ANAActivation.forward`` (``quantlib/algorithms/ana/ana_ops.py:L107``);
* ``ANAUniform.forward`` (``quantlib/algorithms/ana/ana_lib.py:L``);
* ``ana_uniform_cuda.forward -> uniform_forward_cuda`` (``quantlib/algorithms/ana/csrc/uniform_cuda.cpp:L58``);
* ``uniform_forward_cuda_dispatch`` (``quantlib/algorithms/ana/csrc/uniform_cuda_kernel.cu:L145``);
* ``uniform_forward_cuda_kernel`` (``quantlib/algorithms/ana/csrc/uniform_cuda_kernel.cu:L39``).
