.. _systems_package-working_files:

Working files
=============

.. what are working files? why are they important? how should I write them?

**Working files** are Python modules (``.py`` files) and JSON files (``.json`` files) contained in the systems package.
They implement the "blueprints" of the components of DNN learning systems, specify how to choose them and, if needed, how to initialise them.

In particular, the Python modules should implement symbols that will be exposed in the systems namespace, a problem namespace, or a topology namespace.
Instead, the JSON files shoudl specify which symbols to instantiate and, if necessary, the arguments required for such instantiations.
At runtime, QuantLab's experiment manager will:

* import the namespaces into QuantLab libraries;
* import configurations into dictionaries;
* use :ref:`assistants <manager_package-assistants>` to instantiate the actual DNN system components required by the flows.

The next sections describe how to write working files.


Python files
------------

.. what is the purpose of Python files? what is NOT their purpose? where will they be used?
.. which entities MUST be implemented in Python files? which entities CAN be implemented by working files?
.. how should they be structured?

Python working files implement the Python functions and classes that are used by the experiment manager to instantiate the components required to run QuantLab flows.
Python working files should **not** contain information about which symbols exposed in the systems, problem and topology sub-packages should be used to instantiate the components of the learning system.
Python working files should **not** contain information about the actual arguments that should be passed to the chosen functions and the constructor methods of the chosen classes.

In the following, we will describe which symbols are expected to be defined in problem and topology sub-packages.

Pre-processing (problem)
^^^^^^^^^^^^^^^^^^^^^^^^

Zero or more sub-classes of `callable Python objects <https://docs.python.org/3/reference/datamodel.html#object.__call__>`_ that can transform data.
The nature of the input and output data structures is left unspecified; the only requirement is that they don't break pipelines of class :py:class:`torchvision.transforms.Compose` of which they are part.
These objects implement non-structural pre-processing transforms, such as stochastic lighting variations in image data sets or deterministic band-pass filters, that are usually topology-agnostic.

If defined, these symbols must be exposed in the problem namespace.
Therefore, they must be implemented in Python modules contained inside the problem utils sub-package.
They can be imported by topology sub-packages to assemble pre-processing pipelines (actually, this should be their main purpose).

Pre-processing (topology)
^^^^^^^^^^^^^^^^^^^^^^^^^

One or more sub-classes of :py:class:`torchvision.transforms.Compose` implementing pipelines of non-structural (e.g., data augmentation) and structural transforms.
They should return objects of class :py:class:`torch.Tensor` that are compatible with the DNN topology.
Since at least a structural transform of type :py:class`torchvision.transforms.ToTensor` is required to cast the data set-specific data structures to PyTorch tensors, at least one such abstraction must always be defined.

The symbol must be exposed in the topology namespace.
Since it is not uncommon to try several different pre-processing pipelines to improve the performance of the system, we suggest to create a ``preprocessing`` sub-package of the topology sub-package.
Then, for each pre-processing pipeline, we suggest to create a separate Python module to host its implementation.
Finally, export the associated symbol in the preprocessing sub-packages namespace, and from there into the topology namespace.

Data loading (topology)
^^^^^^^^^^^^^^^^^^^^^^

.. todo: this is currently defined in the topology namespace, but it seems it can be made more generic!

Exactly one Python function that outputs objects of class :py:class:`torch.utils.data.Dataset`.
This function must accept as arguments all and only the following:

* a string indicating which data set partition to load (``train``, ``valid``, or ``test``);
* a string representing the path to the folder containing the data points;
* an positive integer representing the number of folds in a cross-validation experiment;
* a non-negative integer representing the ID of the fold for which to prepare the data set;
* a non-negative integer representing the seed for the random number generator; this is used to guarantee consistency of CV splits of the data set in-between interrupted or crashed training runs;
* an object of class :py:class:`torchvision.transforms.Compose`, implementing the pre-processing transforms going from a raw data point to a PyTorch tensor that is an acceptable input for the DNN topology.

The user can define a data set-specific cross-validation split function or, if it is sufficient to ensure the statistical consistency of the CV splits, the default CV split function exposed in the systems namespace.

The symbol **must** be named ``load_data_set``, and must be exposed in the topology namespace.
We suggest to implement it directly inside the ``__init__.py`` module of the topology sub-package.

DNN topology
^^^^^^^^^^^^

Exactly one sub-class of :py:class:`torch.nn.Module` implementing the chosen DNN topology (e.g., AlexNet) or a family of DNN topologies (e.g., VGGs, ResNets).
It will accept objects of class :py:class:`torch.Tensor` (or lists of such objects) as inputs and produce tensors (or lists of such objects) as outputs.
Since this is the very core of a DNN learning system, this abstraction must always be defined.

The symbol must be exposed in the topology namespace.
We suggest to name the symbol in **exactly the same way** as the topology sub-package itself.
We suggest to implement the symbol in a child module of the topology sub-package having the same name as the topology or the topology family, but lower-case (e.g, define the symbol ``VGG`` in the module ``vgg.py``).

Network initialisation
^^^^^^^^^^^^^^^^^^^^^^

Network-specific loss functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Problem metric statistic
^^^^^^^^^^^^^^^^^^^^^^^^

Topology metric statistic
^^^^^^^^^^^^^^^^^^^^^^^^^

Problem-specific statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Topology-specific statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


The last sub-group of components contains the abstractions to perform post-processing.
Therefore, these components are specified in the topology sub-package's ``postprocessing`` sub-sub-package.
There is only one mandatory component: a problem- and topology-specific *task statistic*.
This class is a sub-class of :py:class:`manager.meter.statistics.TaskStatistic`.
This class must include a method ``update`` that takes in input two ``Tensor``s and updates a running trace of the performance of the system with respect to the problem-specific metrics.
A well-defined task metric is usually problem-specific but topology-agnostic.
Therefore, the creator of a new problem sub-package is recommended to implement a *template* for this class.
What is left to the implementer of the topology-specific version of the task metric is implementing two "connector" functions whose purpose is transforming the ``Tensor`` outputs of the network and/or the pre-processed ground-truth labels into problem-specific output data structures.
These connector functions are usually named ``postprocess_pr`` (for predicted labels) and ``postprocess_gt`` (for ground truth labels).
With this approach, it should be sufficient for the implementer of a topology sub-package to just import the topology-agnostic ``TaskStatistic`` and subclass it by passing the two "connector" functions to its constructor.

For some topologies, there are additional components that the user might want to specify.
*Deep supervision* loss functions are the simplest example.
These loss functions are not implemented by default in PyTorch, usually because they work just on a specific topology.
We suggest the user to implement these classes in a module with a meaningful name (e.g., ``loss.py``).
Remember that if you want QuantLab to find the implementations, it is necessary to export the names of these newly-defined objects to the topology sub-package's namespace.


JSON functional configuration
-----------------------------

Where Python working files implement "toolkits" available to QuantLab's experiment manager, JSON working files specify which tools should be used and in which way.
JSON working files should contain information about which symbols exposed in the systems, problem and topology namespaces should be used to instantiate the components of the learning system.
JSON working files should contain information about the actual arguments that should be passed to the chosen functions and to the constructor methods of the chosen classes.

JSON working files are also called *shared configuration files*.
Each topology sub-package stores a permanent copy of a configuration file that the user can edit to specify different configurations of the desired DNN system.
At runtime, QuantLab's configuration flow will create an experiment-specific configuration file starting from the shared version.

The JSON functional configuration file must respect a specific format, encoded in a `JSON schema <https://json-schema.org/>`_.
In the following, we summarise its main sections.

.. todo: create a "lightweight" itemised version of the JSON schema
