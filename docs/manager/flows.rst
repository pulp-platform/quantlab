.. _manager-flows-package:

``flows``
=========

This package implements the services that can be accessed through QuantLab's *façade* file, ``main.py``.

These services have been conceived as models of the main stages that usually compose the development of a machine learning system:

* *configuration*, when the components are selected (e.g., the DNN topology, the loss function, the gradient descent algorithm) and their hyper-parameters are set;
* *training*, when the model is fit to data; at this stage, a scrupulous researcher might want to corroborate the statistical robustness of the investigation by using cross-validation;
* *testing*, when the model is applied to unlabelled data for getting a gist about its performance in the real world;
* *deletion*, in case the experiment's result have already been archived and moved or when the information collected during the experiment are not deemed useful.






The flows can be invoked during a QuantLab session, and they can be configured via command line arguments.
To understand its usage, navigate to QuantLab's home folder and issue ``python main.py --help``.

.. automodule:: manager.flows.__init__
   :members:

.. automodule:: manager.flows.train
   :members:
