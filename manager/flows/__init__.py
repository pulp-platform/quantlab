# 
# __init__.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

"""The flows that can be invoked during a **QuantLab session**.

This module exposes several abstractions that model the high-level
steps of the protocol used when developing a machine learning system.

For example, the standard protocol used when developing a learning
system around a floating-point deep neural network (DNN) goes as
follows:

  1. inspect the details of the computing ``platform``, such as how
     many CPUs and GPUs it has, whether it is a multi-node machine,
     whether it is possible to launch a multi-process MPI run;
  2. ``configure`` the system, e.g., by defining the network topology,
     the pre- and post-processing functions for the data, the batch
     size, the loss function, and the optimisation algorithm;
     considering the information returned by the inspection flow,
     certain hyper-parameters (e.g., the batch size) can be set in
     such a way to minimise the time spent on training and test;
  3. ``train`` the system;
  4. ``test`` the performance of a trained system on validation data;
  5. [optional] ``delete`` the experiment's logs, e.g., when the
     experiment has been wrongly configured and therefore yields no
     information, or when a copy has been stored elsewhere.

Quantized neural networks (QNNs) have attracted much interest in the
last few years. Being a younger research field than DNNs, QNNs pose
more open-ended questions than their floating-point counterparts. For
instance, many training algorithms for QNNs have been proposed, most
of which rely on variants and approximations of the back-propagation
algorithm that are not natively supported by deep learning frameworks
such as TensorFlow or PyTorch. It is common practice in the field to
validate the performance of a new training algorithm by converting
floating-point network topologies that perform well at their tasks
to fake-quantized counterparts. This conversion is attained by
replacing the original framework's native floating-point building
blocks with custom blocks that implement the new algorithm. This
replacement is usually done by hand, rewriting the original
floating-point network using quantized blocks. As most manual
activities, this approach to the conversion problem is error-prone,
making it trickier to reproduce other researchers' results. Another
apparent limitation of most research works on QNNs is the lack of
attention for the deployment problem, that requires converting
trained fake-quantized networks into true-quantized networks that can
be correctly executed on integer arithmetical units or on specialised
accelerators. As an attempt to counteract these two limitations,
QuantLab exposes to the user a ``quantize`` flow that launches an
interactive IPython session where the user can:

  * import QuantLab's quantization tools from the ``quantlib`` package
    and implement the float-to-fake and fake-to-true conversions using
    the abstractions implemented therein;
  * export the defined set of rules to a script that can be stored for
    future reuse or replicability.

The modified protocol for the development of a QNN goes as follows:

  1. [optional] activate the ``quantize`` mode and find a set of rules
     to convert a floating-point network into a fake-quantized
     counterpart; export the defined rules to a script to automate the
     conversion in future runs;
  2. inspect the details of the computing ``platform``, such as how
     many CPUs and GPUs it has, whether it is a multi-node machine,
     whether it is possible to launch a multi-process MPI run;
  3. ``configure`` the system, e.g., by defining the network topology,
     the pre- and post-processing functions for the data, the batch
     size, the loss function, and the optimisation algorithm; also,
     define the hyper-parameters of the quantization algorithm;
     considering the information returned by the inspection flow,
     certain hyper-parameters (e.g., the batch size) can be set in
     such a way to minimise the time spent on training and test;
  4. ``train`` the system;
  5. ``test`` the performance of a trained fake-quantized network on
     validation data;
  6. activate the ``quantize`` mode and find a set of rules to convert
     a trained fake-quantized network into a true-quantized one;
     export the defined rules to a script to automate the conversion
     in future runs;
  7. ``test`` the performance of the true-quantized network on
     validation data;
  8. [optional] ``delete`` the experiment's logs, e.g., when the
     experiment has been wrongly configured and therefore yields no
     information, or when a copy has been stored elsewhere.
"""

from .platform  import platform
from .configure import configure
from .delete    import delete
from .train     import train
from .test      import test
from .quantize  import quantize

