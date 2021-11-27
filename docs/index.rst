.. QuantLab documentation master file, created by
   sphinx-quickstart on Tue Aug  4 17:55:35 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

########
QuantLab
########

QuantLab is a Python tool to explore *deep neural networks* (DNNs).
Due to its support for :abbr:`quantisation-aware training (QAT)` algorithms, it can be particularly useful to develop **quanted neural networks** (QNNs).

The structure of QuantLab is highly modular, designed to meet the needs of different users.
If you are working in a team with multiple colleagues, and often spend considerable time in understanding each other's code to reuse trained models, QuantLab provides experiment management functionalities.
If you are trying to solve a specific problem using a QNN but you have not yet found a suitable training algorithm, you can plug QuantLab's quantisation algorithms into your existing PyTorch codebase.
If you are looking for abstractions to ease the deployment of your QNN to a target device, QuantLab's computational graph editing functionalities might help you.
QuantLab provides device-specific graph optimisation and code generation functionalities for some devices too: in this case, you might want to have a look at the :ref:`list of supported backends <supported_backends>`.

QuantLab fulfills two main roles:

* **experiment manager**, where the abstractions implemented in the :ref:`manager package <manager_package>` act in coordination with the :ref:`systems package <systems_package>` (and its structure of sub-packages) to run experiments and organise their results in a coherent way;
* **quantisation library**: you can find several quantisation algorithms (both *quantisation-aware* and *post-training*), graph editing functionalities, and even graph optimisation and code-generation support for some platforms in the :ref:`quantlib package <quantlib_package>`.

QuantLab is built on top of the `PyTorch <https://pytorch.org/>`_ deep learning framework.
It was developed at the `Integrated Systems Laboratory <https://iis.ee.ethz.ch/>`_ (Institut für Integrierte Systeme, IIS) of ETH Zürich as part of the `PULP project <https://pulp-platform.org/>`_.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   gettingstarted/gettingstarted
   systems/index
   manager/index
   quantlib/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
