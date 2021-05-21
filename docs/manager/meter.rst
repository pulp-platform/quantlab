.. _manager-meter-package:

``meter``
=========

This package implements the software abstractions to track the performance and the inner state of a DNN system.
The set of trackable statistics includes:

* performance statistics related to the **functional purpose** that the system is designed to accomplish; they include *task metrics* and their proxy used in supervised learning, the *loss*;
* performance statistics describing the **computational efficiency** of a training run; these statistics are intended to help users speed up their training runs;
* **descriptive statistics** that can be used to shed light on training inefficiencies, training issues, but also more general properties of DNN topologies.


``statistics``
--------------

Deep learning frameworks such as TensorFlow and PyTorch process and transform array data structures.
In the case of PyTorch DNN systems, the networks are objects of class :py:class:`torch.nn.Module`, whereas the arrays are wrapped into objects of class :py:class:`torch.Tensor`.
QuantLab's experiment manager works under the assumption that all the trackable statistics are computed on top of ``Tensor``s.

PyTorch programs usually define (or at least can have easy access to) symbolic handles for several ``Tensor``s: the parameters of a ``Module``, its inputs, outputs, and auxiliary arrays such as the pre-processed ground-truth labels or the outputs of *deep supervision* nodes.
But not all ``Tensor``s can be accessed as easily.
Differently from TensorFlow, Pytorch builds computational graphs dynamically.
This design choice has the effect of not exposing to the user any symbolic handle (i.e., a Python name) to the arrays that are created and destroyed internally to the network, such as features or gradients.
Conveniently, PyTorch also implements a system for attaching *callbacks* to a network's modules, that are triggered as soon as the forward and backward functions associated with its composing modules are called.
This system relies on the so-called `hooks <https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks>`_.
*Hooks* make it possible for the user to register callbacks that have access to ``Tensor``s processed internally to the network.

For these reasons, QuantLab defines the following dichotomy:

* **callback-free statistics**, that can be computed without the use of *hooks*;
* **callback-based statistics**, that require the use of *hooks* to be computed.

The standard learning algorithm for DNNs is an iterative application of the *backpropagation* (BP) algorithm for gradient computation and of the *gradient descent* (GD) algorithm (or a variant of its) for numerical optimization.
The **iterative** nature of this algorithm implies that the temporal dimension is relevant for DNN systems, that can therefore be considered particular *dynamical systems*.
Being able to "photograph" the state of a DNN system at different moments in time and inspecting its statistics according to their temporal ordering can yield valuable information to understand its properties.

Note that the statistics about a DNN training run can be considered time series.
Therefore, apart from considering *instantaneous statistics*, it is also possible to compute *running statistics* (such as moving averages) by operating on these time series.
This is the second dichotomy that QuantLab defines when defining statistics:

* **instantaneous statistics**, that are computed considering just the state :math:`s_{t}` of the DNN system at a specific iteration :math:`t \in \mathbb{N}_{0} = \{ 0, 1, \dots \}`;
* **running statistics**, that are computed considering multiple states :math:`\{ s_{t_{0}}, \dots, s_{t_{R - 1}} \}` of the DNN system, where :math:`0 \leq t_{0} < \dots < t_{R - 1}`; usually, these states are consecutive, i.e., :math:`t_{r} = t_{r + 1} \,,\, r = 0, \dots, R - 2`.

.. The optimisation of a DNN starts from the available *data set* :math:`\mathcal{D} \,:\, X \times Y \to \mathbb{N}`.
.. The algorithm is stochastic in the composition of the *mini-batches* (also simply called *batches*) of data points that are used to evaluate the *empirical loss* at each iteration.

To conclude this sub-section, we touch on some implementation details of QuantLab statistics that have been conceived in relation to multi-process runs.
QuantLab users might not always want to write the computed statistics to file, but when they do they should write them to `TensorBoard <https://www.tensorflow.org/tensorboard>`_ event files using objects of class :py:class:`torch.utils.tensorboard.SummaryWriter`.
In multi-process applications, if all the processes attempt to write the same file at the same time, the operating system will usually serialise the accesses to the file.
This serialisation will also have the effect that the last process accessing the file will overwrite the information previously written to the file by other processes.
The risk of such a situation is known as a *write-after-write* (WAW) hazard.
QuantLab's experiment manager avoids WAW hazards by ensuring that only a single process (the so-called *master*) can write to disk.

Unfortunately, in multi-process deep learning applications it is usually not the case that the master alone holds all the information required to compute the desired statistics.
Indeed, the main reason to scale DNN training runs to multi-process scenarios is exploiting *data parallelism* to speed up convergence.
In multi-process data parallel training runs, each process holds only a fraction of a specific input, output, feature or gradient array.
Therefore, in those cases where all the pieces of an array partition must be recomposed to ensure consistency of the associated statistic, the non-master processes (the so-called *workers*) must communicate their array pieces to the master.
Only after this synchronisation has happened the master can compute the statistic and write it to disk.
To summarise: each process usually has to participate to the "preparation" of an array before computing the associated statistic, but the actual computation and the write to disk are done by the master only.

Remember from section `platform`_ that QuantLab multi-processing is based on the Horovod package, which is in turn built on top of the *message passing interface* (MPI).
MPI applications spawn multiple processes, each of which executes the same code, but conditioned on a process identifier called the *rank* of the process in the group of spawned processes.
To "homogenise" the creation of statistics objects in a way that ensures that non-master processes can never write to disk, QuantLab introduces the concept of *writer stub*.
A *writer stub* is a plain Python object acting as a wrapper around a null-initialised attribute.
It is the responsibility of the master process to make this attribute point to a ``SummaryWriter`` at run-time.

*Writer stubs* have an additional benefit.
QuantLab's training flow supports cross-validation experiments, and one of its most valuable properties are its results organisation functionalities, where the statistics about the training runs associated with different training folds are stored to different folders.
This functionality requires to destroy the writers associated with a fold at the end of a training run and creating them anew at the beginning of the next one.
*Writer stubs* can "survive" the destruction of the objects they wrap.
Therefore, it is sufficient to create them just once per flow, making it possible for ``assistants`` to reuse them multiple times.
The master process just needs to plug new ``SummaryWriter``s into the correct stubs at the beginning of each run, and destroying them at the end of the run.

``meter``
---------

From a software engineering perspective, QuantLab tracks statistics using an *observer* pattern.

The *subject* of the observations is usually the :py:class:`torch.nn.Module` implementing the target network topology.
This includes its input and output arrays, features, gradients, and also auxiliary arrays such as the pre-processed ground-truth labels, or the outputs of *deep supervision* nodes.
The only supported exception are :py:class:`torch.optim.Optimizer` objects, that can be observed to monitor the learning rate used during training.

Objects of class ``Meter`` act as *mediators* of this observer pattern.
At each iteration of a training run, the *mediator* signals four events to the *observers*:

* *step change*: the batch identifier has changed;
* *forward pass start*: the inputs are being prepared, but they have not yet been passed through the network;
* *forward pass end/backward pass start*: the outputs (including the loss) have been computed; the network is ready for gradient computation and optimisation;
* *backward pass end*: the gradients have been computed and the optimisation step has been performed.

Some examples of how the events can be used by different classes of statistics:

* *step change*: the state of each *observer* is updated accordingly; this information can be used by instantaneous, callback-based statistics to determine whether to register hooks or not; running callback-free statistic can reset their state at the beginning of a new epoch;
* *forward pass start*: callback-based statistics can register their hooks; callback-free statistics can create a temporary copy of the parameters of the system to compare them with their counterparts after the pending step;
* *forward pass start/backward pass start*: running callback-free statistics can be updated;
* *backward pass end*: callback-based statistics can remove their hooks; callback-free statistics can compare the new parameters of the system with the old ones.
