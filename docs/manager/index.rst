.. _manager_package:

The ``manager`` package
=======================

This section introduces the ``manager`` package, and is organised as follows:

* the :ref:`first sub-section <manager_package-improving_systems>` is a motivational discussion about structuring the analysis of machine learning systems, and in particular DNN systems;
* the :ref:`second sub-section <manager_package-statistical_laboratory>` is also a motivational discussion introducing the basic concepts of *design of experiments* and their influence on the structure of the package;
* the :ref:`third sub-section <manager_package-overview>` provides an overview of the package and contains references to its documentation.


.. _manager_package-improving_systems:

Improving the quality of engineered systems
-------------------------------------------

A **system** is a collection of multiple components whose interactions originate a higher-level behaviour.
From an epistemological point of view, we can distinguish two classes of systems:

* *natural* systems: the nature of the components and their mutual interactions might be unknown; it might be unclear how to control the components, their interactions, or even the overall system;
* *engineered* (i.e., artificial) systems: the components and their interactions are explicitly specified at design time; usually, they can be controlled (e.g., using configurable parameters).

Machine learning systems are engineered systems.
With respect to other engineered systems, we usually have limited or no knowledge of the data distribution to which a machine learning system will be exposed when deployed in the real world.
Consequently, our best strategy to develop a good machine learning system is identifying the components and the corresponding configurations that provide the best performance in a controlled environment.

A DNN-based learning system has multiple components:

* the pre-processing functions;
* the DNN topology;
* the loss function;
* the learning algorithm (comprising both the gradient descent algorithm and the learning rate scheduling algorithm);
* the post-processing functions.

In turn, most of these components can be configured in multiple ways.
For instance:

* DNN topologies from the same topology family might differ by number of layers;
* there might be several candidates for the loss function;
* there are several variants of the gradient descent algorithm.

How can we assess the impact of different components and different configurations on the system's performance?
Can we do this following a methodologically sound approach?


.. _manager_package-statistical_laboratory:

Envisioning a statistical laboratory
------------------------------------

The field of `design of experiments (DoE) <https://en.wikipedia.org/wiki/Design_of_experiments>`_ was born to develop statistically-grounded tools and methodologies to quantify the impact of controllable variables on a target system.
When designing QuantLab's experiment management capabilities, we took inspiration from DoE, and also borrowed some of its terminology:

* *population*: a collection of systems about which we would like to draw some inference;
* *individual*: a member of the population, possibly sampled at random;
* *treatment*: the instance of a variable encoding the different actions that can be taken over an individual; its value can be set by the experiment designer;
* *response*: the random variable encoding the state and the performance of an individual who has been administered a given treatment;
* *measurement*: a realisation of the response;
* *experimental unit*: an abstraction pairing an individual with a treatment;
* *experimental run*: the process by which a treatment in a given an experimental unit is administered to the corresponding individual; measurements are usually collected for a limited time span after the administration has occurred;
* *experimental design*: a (finite) collection of experimental units, usually differing by treatment, individual, or both; the measurements of the corresponding experimental runs can be compared using statistical tools to infer conclusions about the response of a more general population of individuals to a specific treatment or collection of treatments.

"An experiment manager at a statistical laboratory wants to measure the reponse of an individual to a specific treatment.
As a first step, the manager registers all the details about the experimental unit on a **configuration document**, and she stores this document in a **logbook**.
Since an experimental run might entail several different **flows**, the laboratory offers dedicated services for each flow.
When starting a flow, the manager hands specific pieces of information stored in the logbook to dedicated **assistants**, so that each of them can assemble a **component** required to run the flow.
In particular, the manager instructs a dedicated assistant to prepare the **measurement intstruments** required to track the state of the individual throughout the experimental run, recording informative statistics and measuring its performance.
The flow can begin."


.. _manager_package-overview:

Overview
--------

Thisj package implements the components that enable QuantLab's experiment management functionalities.
It is structured to mirror the aforementioned vision of a statistical laboratory.

QuantLab's experiment management functionalities are intended to assist the exploration of DNN-based learning systems.
QuantLab aims at providing this assistance by:

* increasing the time available for functional and algorithmic explorations by reducing the time invested in setting performance-related and other engineering details;
* structuring the results of the experiments in such a way to ease the following statistical analysis.

In its role of experiment manager, QuantLab is a command line interface (CLI) tool.
QuantLab's services are called *flows*, and can be accessed through the *façade* file ``main.py``.
Flows are implemented in the :ref:`flows sub-package <manager-flows-package>`.
The development of a machine learning system usually includes both:

* **functional** aspects, such as the exact system implemented (e.g., a support vector machine, a random forest, a deep neural network), its components and hyper-parameters (e.g., the loss function, the gradient descent algorithm used for optimisation, the batch size), and its task accuracy;
* **engineering** and **performance** aspects, such as support for parallel hardware, multi-process synchronisation, and management of disk transactions.

The :ref:`platform sub-package <manager-platform-package>` abstracts away the platform-specific details that are relevant for the performance of experiments, but not relevant for the functionality of the associated DNN system.
Whereas the :ref:`systems package <systems-package>` implements the "blueprints" of the components of different DNN systems and exposes a prototype configuration file, QuantLab flows instantiate such "blueprints" at run-time using *assistants*, abstractions implemented in the :ref:`assistants sub-package <manager-assistants-package>`.
This assembly is orchestrated by *logbooks*, abstractions implemented in the :ref:`logbook sub-package <manager-logbook-package>`.
This sub-package also implements disk interaction functionalities.
Since QuantLab has been designed with a focus on research, the capability of tracking the evolution of DNN systems and their state during training runs is critical to its purpose.
The :ref:`meter sub-package <manager-meter-package>` implements several abstractions to provide this functionality.

.. toctree::
   :maxdepth: 1
   :caption: Sub-packages:

   flows
   platform
   assistants
   logbook
   meter
