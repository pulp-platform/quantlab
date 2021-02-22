.. QuantLab documentation master file, created by
   sphinx-quickstart on Tue Aug  4 17:55:35 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

QuantLab
========

**QuantLab** is a Python framework designed to explore quantized neural networks.
It allows to design and compare multiple training algorithms, organise experimental results, facilitate knowledge transfer and also deploy trained models on multiple hardware platforms.
QuantLab is built on top of the `PyTorch`_ deep learning framework.

.. _PyTorch: https://pytorch.org/

The structure of QuantLab is highly modular, designed to meet the needs of different users.
If you are looking for a flexible quantization library to train a QNN on a specific application, you can just plug the graph editing and quantization operations into your existing PyTorch codebase.
Then, you could use one of QuantLab's backends to deploy your trained model on the target device.
If you are working in a team with multiple colleagues, and often spend considerable time in understanding each other's code to reuse trained models, QuantLab also provides organising code functionalities.
From coherent problem containerisation, through the organisation of experimental results, to the generation of deployment-ready models, QuantLab encourages the use of a common structure in your projects, covering all their aspects, from exploratory data analysis to model deployment.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quantlib/index
   backends/index
   problems/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
