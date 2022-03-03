.. _quantlib-package:

A quantization library
======================

The QuantLib package is the heart of QuantLab.
It consists of three sub-packages:

* the :ref:`algorithms sub-package <quantlib-algorithms-package>` contains the implementations of learning algorithms for QNNs;
* the :ref:`editing sub-package <quantlib-editing-package>` contains the implementations of graph analysis, editing, and morphing functionalities;
* the :ref:`backends sub-package <quantlib-backends-package>` contains the implementations of graph optimisation and code generation functionalities for specific devices.

Before giving an overview of the structure of this package and its abstractions, we must define some terminology.
.. todo: We call a *floating-point graph* a PyTorch :py:class:`Module`...


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   algorithms/index
   editing/index
   backends/index
