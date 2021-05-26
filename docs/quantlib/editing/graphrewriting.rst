.. _quantlib-graphs-graphrewriting:

Rewriting computational graphs
==============================

Converting fake-quantized computational graphs into true-quantized computational graphs requires two sets of functionalities:

* *graph rewriting*, where specific sub-graphs of the source graph are replaced with new sub-graphs;
* *folding* and *casting*, arithmetic operations whose nature is strictly related to digital arithmetic, and might even be device-specific.

Computational graphs are not general graphs; indeed, they are attached a very specific semantic.
Computational graphs are directed, bipartite graphs whose nodes represent operands and operations on these operands, and whose arcs represent dependency relationships between operands and operations.
This clarification is important in that graph rewriting can be explained without reference to arithmetic aspects.
Nevertheless, understanding the graph rewriting rules applied in the context of fake-to-true conversion mandates that the reader has also a firm grasp on the principles of *number representations* and *digital arithmetic*.

For this reason, this section of the documentation is structured as follows:

* the first sub-section will introduce the approach to algebraic graph rewriting without referring to arithmetic notions;
* the second sub-section will summarise the principles of *number representations* and *digital arithmetic*;
* the last sub-section will illustrate some of the general-purpose graph rewriting rules implemented in QuantLab's quantlib sub-package.

Due to the often device-specific nature of the required folding and casting operations, we refer the reader to the examples of arithmetic computational graph rewriting rules implemented in the :ref:`TWN accelerator <backends-twn-accelerator>` backend package.


The algebraic approach to graph rewriting
-----------------------------------------

Graph rewriting is a topic in graph theory concerned with the transformation of graphs into other graphs.
To qualify as a *rewriting*, such transformations must be expressed as a series of intermediate, well-defined steps.
Such a step (or, recursively, a sequence of such steps) is called a *derivation*.
The input to a derivation is called its *source graph*, whereas its output is called its *target graph*.

.. We refer the interested reader to the paper * `Introduction to graph grammars with application to semantic networks <Ehrig1992>`_ * as it is a sufficiently complete reference to understand the applications of graph rewriting in QuantLab.
   It is interesting to note that the `original works <Ehrig1973>`_ on graph grammars date back to the 1970s, i.e., the same period during which the *dataflow* paradigm underlying deep learning frameworks such as PyTorch and TensorFlow.
   .. _Ehrig1992: https://pdf.sciencedirectassets.com/271503/1-s2.0-S0898122100X03045/1-s2.0-089812219290124Z/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEPj%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDvRCLkLWYBco53XI7TcsDECZ4kd0RAIdXSqtnkquZm0QIgY%2FYcgzh9l9sCjgV10GPAjNhAIUrAwfoE0yil4Zm3aegqvQMIkf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARADGgwwNTkwMDM1NDY4NjUiDCIBRJ1Y%2FU2w7JQ21CqRA%2FTxk8NhSW7mEsJTVb4MaG78hn4kI491RpsJGEeEqltLmdwPGutPi03zxd9lIOggPhn8FZuX9NVrXJexXvrIrF1xfnMMnyjq54rHZd0Py5VJSYPBfMFjHsM7gGROXsOQDz5ZeD1Nr%2BgDBfSPTaImj%2FTxYB9roUrYXSQtAtHdX7lvtWHnCPhiWRzHSmcmWjkMZ5SzCmI%2BZCGKylI0ZWSQqT9AuwjsEh1nkWiOJ%2BZyaVab0CizMM31OsfDFe8K%2FDiskWWpPeL86aN6o0d81ckW%2FQFfP1tAB2SA34txrLjAV9VCBaY1wIvAdi0SXJuB9PKxZq2J0D8pEwP1xI6sp%2B6a63KctXHj7lHZjpM7WR2fYFfrNZD3JO4%2Bl4nLT9fDesKeJoitgknKpvBZGhkQLH9U7ik6e0pW04P9A0Sv2xSPadG0LYXJlGzwHB5pYc8pkuuvVqcZ2gGiR1xPaAvD7FHZyQeoJFWbixPFBGQ45jWdeDEDNnDcrdA4GyU4RskQLAGaTspnF7sZXNdtNaRE4fW0D6ThMJCOmoUGOusBi3UIjsHfVG%2FP8UCCCk85ZnK0ZbbL1c5gmKu%2B%2Fun9tsMOsp0tiYU7v2X%2FHGk3gFUBAqMdUUgGGNMSGLFMi75fJLV5LvPRgwrBPDO1mAboBhZMDG3spoyFXTAro%2B%2FMFEYNpavKD%2BjVJPfwm%2ByOI1z6xUcJ9zLkqDSkuvasYRWjueprn7MbuF56I%2BM0gizxmPul5VF5aDv6KTsxuwDClcTpZcmVrTZjMe%2BSkdhgLjJR4hARnN3zvO8SeAJd78kOVMFi2Jh2q9SsotyilIt2aIdCWISgGaobXgVg6KaVS%2F88XMJxCgY%2FvZFGt2jDwg%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20210520T162043Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYRLTKBHC2%2F20210520%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=d0df1784c0e0fff6f74a99bc3882d8a866845c3af46ab3c0fe4a7009a45b008b&hash=4665f65c174ff1a9096585a64d0219cdba254cebe3a3b54d95cf3c11cbe70f46&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=089812219290124Z&tid=spdf-62cf739e-b552-47cf-9cb6-5bd46c3992b6&sid=fd70d64d4be99943869bf630895f4e29c0a4gxrqb&type=client
   .. _Ehrig1973: https://ieeexplore.ieee.org/document/4569741


Principles of number representations and digital arithmetic
-----------------------------------------------------------


