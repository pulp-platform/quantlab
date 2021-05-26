.. _quantlib-algorithms-package:

Learning algorithms for QNNs
============================

Quantized neural networks (QNNs) are not a novelty in artificial neural network research.
Indeed, the original perceptron consisted of a first layer of (non-trainable) neurons using ternary weights, and its activation function was discontinuous.
The difficulty of deriving proper *error-correcting* signals to update the parameters of the model led research in multi-layer perceptrons to a stall.
This situation was solved only with the introduction of differentiable activation functions, which opened the way to the backpropagation algorithms and gradient-based learning.

The importance of QNNs has been revived in recent years, due to the possible benefits in terms of reduced memory footprint and more energy-efficient arithmetic.
This idea has given birth to a vital research field, with many algorithms being proposed.
The purpose of this library is to provide researchers with a rich pool of learning algorithms for QNNs.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ste
   inq
   ana
