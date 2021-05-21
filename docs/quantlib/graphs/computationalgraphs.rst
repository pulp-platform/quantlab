Computational graphs
====================

A computational graph is a directed, bipartite graph.
The nodes in a partition are arrays representing the data being processed, whereas the nodes in the other partition are the operations that process the data.
Arcs (i.e., directed edges) represent input/output relationships.

QuantLab deals with two types of computational graphs:

* `ONNX`_ graphs, that are reconstructed from PyTorch :py:class:`Module`s using PyTorch's JIT module tracing functionalities;
* PyTorch graphs; an experienced user might be aware of the fact that PyTorch graphs are dynamically built at run-time (differently from TensorFlow, they are not statically specified): QuantLab retrieves the connectivity between PyTorch modules by analysing the traced graph.

Fake-to-true conversions are based on the algebraic approach to graph rewriting explained in the section of the docs dedicated to :ref:`graph rewriting rules <quantlib-graphs-graphrewriting>`_.

.. _ONNX: `onnx-home`_
.. _onnx-home: https://onnx.ai/
