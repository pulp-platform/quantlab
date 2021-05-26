.. _quantlib-backends-package:

``backends``
============

If you got here, you have hopefully trained your QNN successfully, and converted it to a valid *true-quantized* format.
If this is the case, then it is probably time to deploy your model to a target hardware device that can execute it.

The deployment stage spans the steps going from an abstract representation of your model (usually an ONNX graph) to executing machine code on the target device.
These steps usually include:

* *graph optimisation*, the process through which the nodes in an ONNX graph (or any other low-level computational graph) are reordered, fused, or otherwise transformed while preserving functional consistency; the purpose of graph optimisation is exploiting device-specific hardware properties such as its memory hierarchy;
* *code generation*, the process through which the optimised ONNX graph is transformed into a sequence of C/C++ files implementing its operations;
* *compilation*, the process through which the C/C++ files are transformed into executable machine code for the target device; this step usually includes code optimisation as a sub-step;
* *execution*, the process through which your QNN program is running on the target device and processing real-world data points.

The backends package consists of multiple *backend sub-packages* whose purpose is performing the first two steps: graph optimisation and code generation.
Compilation and execution should be taken care of using device-specific tools.


.. _supported-backends:

.. toctree::
   :maxdepth: 2
   :caption: Supported backends:

   twnaccelerator/index
