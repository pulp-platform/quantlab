``backends``
============

If you got here, you have hopefully trained your QNN successfully, and
converted it to a valid true-quantized format. It's time to deploy your
model to a target hardware platform which can execute it!

Usually, the neural networks produced by QuantLab will have been trained using
parallel floating-point hardware like GPUs, where the operations involving
quantized features (if not also the parameters!) are just mimicked using
floating-point arithmetic. We therefore call the corresponding graphs
**fake-quantized** graphs. To execute these models on specialised accelerators
(e.g., chips supporting dedicated ISA extensions for low-precision integer
operands), we need to convert fake-quantized graphs into **true-quantized**
graphs.


Folding and casting
-------------------

To understand how these graph transformations work, we first recap the basics
of **data formats** and **digital arithmetic**.

A data format is defined by two properties: the **data type** (signed integer,
unsigned integer, signed fixed-point, unsigned fixed-point, floating-point)
and the **precision** (64, 32, 16, 8 or even a lower number of bits). The
process of converting from a data format to another is called **casting**.

**Folding** is the process or reordering the floating-point operations that
occur in the context of a fake-quantized layer in order to separate the
integer part of the computation from the part of the computation involving
fractional operands (either fixed- or floating-point numbers).

Casting is the process of converting the fixed- and floating-point operands
computed during folding to data formats which can be processed by the target
hardware platform.

.. warning:: Describe the example for the TWN accelerator: introduce
   accumulators, signed vs. unsigned operands (flip weights and gammas for
   negative gammas), probabilistic analysis of errors.
