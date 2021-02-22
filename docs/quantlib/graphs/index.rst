``graphs``
==========

The most discouraging aspects of the research on quantized neural networks (QNNs) are faulty reproducibility and portability.

Faulty reproducibility issues arise mostly from two facts:
* the inventors of a new algorithm implement custom operators and optimisers starting from those provided by their favourite frameworks (Caffe, TensorFlow, PyTorch);
* the inventors create *ad hoc* versions of popular benchmark networks (usually convolutional neural networks such as AlexNet, VGG, ResNets) by replacing the original full-precision blocks with the quantized counterparts previously developed.
When an algorithm is ported from a source framework to a target framework, the researcher needs to reimplement the original structures and operators on top of the different elementary building blocks provided by the target framework.

By *faulty portability* we refer to the fact that certain implementations, although seemingly effective on paper, do not take into account the transformations required to turn the network trained in software into a model which can actually run on energy-efficient hardware.
Indeed, whereas deep learning frameworks are built around floating-point arithmetic, either in double-, full- or half-precision (i.e., 64-, 32- or 16-bit), QNNs should actually use low-bitwidth integer operands.
Since this deployment goal is the essential motivation of QNNs, it is crucial that this conversion is done correctly; possibly, the feasibility of such a conversion should be assessed even before starting training the network with the proposed algorithm.

To try and counteract these issues, the ``graphs`` sub-package provides a library of tools to:
* analyse PyTorch networks (e.g., display the connectivity structure of a network's layers);
* edit PyTorch networks (e.g., replace a specific set of convolutional layers with equivalent operators which can quantize the weights at training time);
* morph *fake-quantized* PyTorch networks into *true-quantized* PyTorch networks, i.e., produce networks which use only integer arithmetic.
In particular, the last functionality is based on the fact that it is possible to **completely** embed integers in a given precision into the set of floating-point numbers in a higher precision.
Certain operations which are available in floating-point arithmetic (such as *addition*, *multiplication*, *flooring*, *rounding*, *ceiling*) can be implemented efficiently in integer arithmetic or specialised datapaths.
Instead, other operations (such as the *division* applied by batch normalisation layers) should be completely absent in a true-quantized graph.
True-quantized network should rely only on operands and operations which can be actually implemented in hardware.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   analysis
   editing
   morphing
