.. _quantlib-editing-package:

Rationale
=========

The most discouraging aspects of the research on quantized neural networks (QNNs) are faulty reproducibility and deployability.

Faulty reproducibility issues arise mostly from two facts:
* the inventors of a new algorithm implement custom operators and optimisers starting from those provided by their favourite frameworks (Caffe, TensorFlow, PyTorch);
* the inventors create *ad hoc* versions of popular benchmark networks (usually convolutional neural networks such as AlexNet, VGG, ResNets) by replacing the original full-precision blocks with the quantized counterparts previously developed.
When an algorithm is ported from a source framework to a target framework, the researcher needs to reimplement the original structures and operators on top of the different elementary building blocks provided by the target framework.

By *faulty deployability* we refer to the fact that certain implementations, although seemingly effective on paper, do not take into account the transformations required to turn the network trained in software into a model which can actually run on energy-efficient hardware.
Indeed, whereas deep learning frameworks are built around floating-point arithmetic, either in double-, full- or half-precision (i.e., 64-, 32- or 16-bit), QNNs should actually use low-bitwidth integer operands.
Since this deployment goal is the essential motivation of QNNs, it is crucial that this conversion is done correctly; possibly, the feasibility of such a conversion should be assessed even before starting training the network with the proposed algorithm.

To try and counteract these issues, the graphs sub-package provides a library of tools to:

* analyse PyTorch networks (e.g., display the connectivity structure of a network's layers);
* edit PyTorch networks (e.g., replace a specific set of convolutional layers with equivalent operators which can quantize the weights at training time);
* morph *fake-quantized* PyTorch networks into *true-quantized* PyTorch networks, i.e., produce networks which use only integer arithmetic.

In particular, the last functionality is based on the fact that it is possible to **completely** embed integers in a given precision into the set of floating-point numbers in a higher precision.
Certain operations which are available in floating-point arithmetic (such as *addition*, *multiplication*, *flooring*, *rounding*, *ceiling*) can be implemented efficiently in integer arithmetic or specialised datapaths.
Instead, other operations (such as the *division* applied by batch normalisation layers) should be completely absent in a true-quantized graph.
True-quantized network should rely only on operands and operations which can be actually implemented in hardware.

The user flow that we envision is the following:

* pull a floating-point PyTorch model from a public network repository (e.g., GitHub) or a local repository (e.g., a helpful colleague's workspace);
* replace its composing :py:class:`Module`s with counterparts that support fake-quantization training algorithms; there are two assumptions about this step:

  1. that these replacements are mostly involving 1-to-1 swapping of a floating-point node with a fake-quantized counterpart;
  2. that the ``Controller``s associated to the required quantization algorithms are implemented;

  under these assumptions, it is not necessary to perform complex tracing and/or algebraic graph rewriting operations: in fact, it is sufficient to exploit tree traversal algorithms (PyTorch represents DNNs as trees of ``Module``s);
* train or fine-tune the fake-quantized network using the chosen/supported fake-quantization training algorithm;
* trace the fake-quantized graph and rewrite it applying (possibly backend-specific) arithmetic reorderings and algorithmic optimisations.


Computational graphs
--------------------

A computational graph is a directed, bipartite graph.
The nodes in a partition are arrays representing the data being processed, whereas the nodes in the other partition are the operations that process the data.
Arcs (i.e., directed edges) represent input/output relationships.

QuantLab deals with two types of computational graphs:

* `ONNX`_ graphs, that are reconstructed from PyTorch :py:class:`Module`s using PyTorch's JIT module tracing functionalities;
* PyTorch graphs; an experienced user might be aware of the fact that PyTorch graphs are dynamically built at run-time (differently from TensorFlow, they are not statically specified): QuantLab retrieves the connectivity between PyTorch modules by analysing the traced graph.

Fake-to-true conversions are based on the algebraic approach to graph rewriting explained in the section of the docs dedicated to :ref:`graph rewriting rules <quantlib-graphs-graphrewriting>`_.

.. _ONNX: `onnx-home`_
.. _onnx-home: https://onnx.ai/


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   graphrewriting
   lightweight
