.. _systems_package-example:

Solving the CIFAR-10 problem using VGG networks
===============================================

This section describes an example *problem sub-package*, including an example *topology sub-package*; it is organised as follows:

* the :ref:`first sub-section <systems_package-example-layout>` describes how the ``CIFAR10`` problem sub-package and its ``VGG`` topology sub-package shipped with QuantLab are structured; in particular it describes which entities are defined in which files, and how they relate to the functional structure described in the :ref:`related section <systems_package-functional_description>` of the documentation;
* the :ref:`second sub-section <systems_package-example-pythonapi>` documents the Python software abstractions implemented in these packages.


.. _systems_package-example-layout:

Instantiating a DNN system
--------------------------

.. figure:: ./figures/05_data_set.png
   :align: center

.. figure:: ./figures/06_pre-processing.png
   :align: center

.. figure:: ./figures/07_dnn_topology.png
   :align: center

.. figure:: ./figures/08_problem_metric.png
   :align: center

.. figure:: ./figures/09_post-processing.png
   :align: center

.. figure:: ./figures/10_topology_metric.png
   :align: center

.. figure:: ./figures/11_loss.png
   :align: center

.. figure:: ./figures/12_dnn_system.png
   :align: center


.. _systems_package-example-pythonapi:

Python API of the ``CIFAR-10`` and ``VGG`` packages
---------------------------------------------------
