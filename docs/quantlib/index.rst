.. _quantlib-package:

``quantlib``
============

The quantlib package is the heart of QuantLab.
This package contains two sub-packages:

* the :ref:`algorithms sub-package <quantlib-algorithms-package>`, that contains the implementations several quantization algorithms (both quantization-aware and post-training);
* the :ref:`graphs sub-package <quantlib-graphs-package>`, that contains the implementations of graph analysis, editing, and morphing functionalities.

Before giving an overview of the structure of this package and its abstractions, we must define some terminology.
We call a *floating-point graph* a PyTorch :py:class:`Module`



.. toctree::
   :maxdepth: 2
   :caption: Contents:

   algorithms/index
   graphs/index
