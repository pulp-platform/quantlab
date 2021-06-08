Working files
=============

To use QuantLab as an experiment manager, you need to describe the components of the target ML system according to a specific format.
There are two kind of components:

* Python software abstractions;
* JSON functional configuration of the abstractions.

The purpose of the JSON functional configuration file is specifying how to instantiate software abstractions, as well as adding some experiment-specific details.


Python software abstractions
----------------------------

Software abstractions include Python functions and Python classes.
These software abstractions can be grouped in three main sub-groups:

* data loading and pre-processing;
* deep neural network topology;
* post-processing operations.

Data loading components are specified in the topology sub-package's ``preprocessing`` sub-sub-package.
These components must include:
* at least one Python function that outputs objects of class :py:class:`torchvision.transforms.Compose`, implementing pipelines of pre-processing transforms; be sure that the objects of class :py:class:`torch.Tensor` which are output by the transform are compatible with the topology (see next paragraph);
* one function that outputs objects of class :py:class:`torch.utils.data.Dataset`; this function must be given the name ``dataset_load``, and must support a ``train`` Boolean flag dictating whether to return the training data set or the validation data set for your problem.
Optionally, the user can specify additional functions to perform the splits required to run a cross-validation experiment.
Such functions must take as input a list of integers, the number of folds, and the index of the fold that will play the role of the validation data set, and return two lists of integers.
Using these lists, QuantLab's ``DataAssistant`` will split the training ``Dataset`` into two ``Subset``s.
If the user does not provide one, QuantLab will use a default splitting function.
Beware that this default function is not apt for splitting data sets where the data points are not IID (e.g., biomedical data sets where sending observations from the same patient into both the training and validation sets might bias the validation performance of the model.
These symbols should be exported to the namespace

The DNN component represents a DNN topology (e.g., AlexNet) or a family of DNN topologies (e.g., VGGs, ResNets).
It must be a sub-class of the class :py:class:`torch.nn.Module`.
The only mandatory property that this class should satisfy is that its name is exposed in the Python namespace associated with the topology sub-package.
We suggest the user to follow a couple of conventions when implementing this class, whose purpose is to ensure consistency of the codebase and ease navigation of different projects.
In our baselines, we gave the named the class in **exactly the same way** as the topology sub-package itself, to stress that this is the central object of the sub-pakcage.
Moreover, we defined the class in a module whose name is the lower-case version of the class's and the sub-package's name.
We remark that these conventions are optional, although they are very beneficial to ease the understanding of new topology sub-packages.

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

The JSON functional configuration file must respect a specific format, encoded in a `JSON Schema <https://json-schema.org/>`_.
