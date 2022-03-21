# The *additive noise annealing* algorithm
This document describes how to extend a QuantLab configuration to train a QNN using the *additive noise annealing* (ANA) algorithm.


## QuantLab configurations
QuantLab users describe DNN-based learning systems using [JavaScript Object Notation (JSON)](https://www.json.org/json-en.html) files.
JSON files are dictionary-like structures.
For each key-value pair, the key component identifies an entity, whereas the value component describes such an entity.
Since each value component can be a dictionary, JSON allows for entity descriptions of arbitrary depth.

### Basic QuantLab configurations
QuantLab configurations have four top-level key-value pairs, which we call **sections**:
* ``data``, describing how to prepare the data used to train and evaluate the DNN-based learning system; this section includes the description of pre-processing steps (since pre-processing functions are not trainable, pre-processing is semantically distinct from the core problem solved by the target DNN);
* ``network``, describing the network architecture which should be trained, including its topological and computational hyper-parameters;
* ``training``, describing the dynamics of the DNN at training time (loss landscape, gradient descent algorithm, evolution of the hyper-parameters of the learning algorithm);
* ``meter``, describing what should be observed and tracked during training using QuantLab's system of ``Meter``s and ``Statistic``s; this section also includes post-processing information (since post-processing functions are not trainable, post-processing is semantically distinct from the core problem solved by the target DNN).

### Quantising DNNs with QuantLab
Quantisation algorithms act on QNNs at two levels:
* at the **structural level**, they specify the precision of each operation (including both its inputs and outputs);
* at the **training dynamics level**, they describe how quantisation is achieved; information about the training dynamics is relevant for all the quantisation algorithms that require at least a few iterations of fine-tuning.

Therefore, QuantLab configurations describing floating-point DNNs must be extended to include two special ``quantize`` sub-sections to describe a quantisation algorithm:
* the ``network:quantize`` sub-section must describe the structural level of the chosen QAT algorithm;
* the ``training:quantize`` sub-section must describe the training dynamics of the chosen QAT algorithm; for instance, if the QAT algorithm uses hyper-parameters, this section should describe how they evolve through the training or fine-tuning iterations.

The ``network:quantize`` section consists of two sub-sections:
* ``function`` is a string specifying the name of a Python function which takes in input the network object specified according to the ``network`` section and returns a *fake-quantised* (FQ) network; i.e., this function performs a so-called *float-to-fake* (F2F) conversion;
* ``kwargs`` is a dictionary-like JSON object describing the arguments of the F2F conversion function (e.g., the precision of the FQ ``nn.Module``s).

The ``training:quantize`` section consists of two sub-sections:
* ``function`` is a string specifying the name of a Python function which takes in input an FQ network and returns a Python list of QuantLib ``Controller``s that govern the dynamics of the quantisation algorithm;
* ``kwargs`` is a dictionary-like JSON object describing the arguments to the ``Controller``s that specify the training dynamics of the FQ network.


## ANA hyper-parameters

### Float-to-fake conversion
ANA F2F conversion functions must, at some point, create ``ANAModule`` objects.
Each ``ANAModule`` requires six arguments:
* ``nbits`` (integer): the number of bits sufficient to represent each component of the array for which the ``ANAModule`` is responsible (weights for linear operations, features for activation operations);
* ``signed`` (Boolean): whether the integer data type targetted by quantisation is signed or not;
* ``balanced`` (Boolean): if the target integer data type is signed, whether to use sign-magnitude or two's complement notation; in the sign-magnitude notation, the most significant bit is used to represent the sign and zero has two representations;
* ``eps`` (floating-point): the scale of the quantiser;
* ``noise_type`` (string): the parametric family of noise distributions to regularise the quantiser; it resolves into a value from an enumerated (``uniform``, ``triangular``, ``logistic``, ``normal``);
* ``strategy`` (string): the algorithm used to evaluate the output of a quantiser under the chosen noise distribution; it resolves into a value from an enumerated (``expectation``, ``random``, ``mode``).

### ANA training
ANA was conceived to generalise the *straight-through estimator* (STE).
ANA uses additive noise to regularise quantisers and enable gradient descent.
Differently from STE, the regularising noise can change as training progresses.
The noise evolution is governed by ``ANATimer`` objects, each of which requires three arguments:
* a list of ``ANAModule``s, each of which has a regularising noise distribution; all the ``ANAModule``s linked to a given ``ANATimer`` will share the same noise distribution, hence sharing its dynamics;
* a description of the evolution of the mean of this noise distribution;
* a description of the evolution of the standard deviation of this noise distribution.

When using non-static noise distributions, ANA aims at annealing them to Dirac's deltas centred at zero.
This goal can be achieved by ensuring that both the mean and the standard deviation of the regularising noise distributions converge to zero as training progresses.
The evolution of each governing hyper-parameter (mean, standard deviation) is described by three arguments:
* ``fun`` (string): whether the hyper-parameter decay happens over a bounded or an unbounded sequence of epochs; we refer to the sequence of epochs as a *window* or (like in the ANA paper) a *decay interval*; we allow for two options:
  * bounded window size (``bws``);
  * unbounded window size (``uws``);
* ``kwargs`` (dictionary): a description of the decay interval and the speed of decay over the decay interval:
  * ``tstart`` (integer) is the non-negative epoch identifier at which the decay starts (the parameter value is one up to this moment);
  * ``tend`` (integer) is the non-negative epoch identifier at which the decay ends (the parameter value is zero from this moment onwards); note that this value does not need to be defined in conjunction with the ``uws`` value;
  * ``alpha`` is the exponent of the power law regulating the speed of decay;
* ``beta`` (floating-point): normally, the hyper-parameter decays from one to zero; this argument is a non-negative constant multiplying the base value of the hyper-parameter, and you can set it to zero to signal the intention to use a static hyper-parameter.
