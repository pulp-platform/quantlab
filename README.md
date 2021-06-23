# QuantLab

QuantLab is a tool to train, compare and deploy quantized neural networks (QNNs).
It was developed on top of the  [PyTorch](https://pytorch.org/) deep learning framework, and it is a purely-command-line-based tool.

QuantLab consists of two main components:
* a **library** of quantization tools (`quantlib`);
* **organising software** to manage machine learning (ML) experiments (`systems` and `manager` packages, as well as the `main.py` façade script).

QuantLab and `quantlib` have been developed at the [Integrated Systems Laboratory](https://iis.ee.ethz.ch/) (IIS) of ETH Zürich.


## Installation and usage

In the following, we assume that you have cloned the repository to your local machine at your home folder, which we will denote by `~`.
We will conventionally use the UNIX separator `/` when writing filesystem paths (Windows users might have to replace it with the backslash) and the dollar sign `$` to denote the command line prompt.

We call `~/QuantLab` the *QuantLab home*.

### Create an Anaconda environment and install `quantlib`
As a first thing, navigate to the QuantLab home and use [Anaconda](https://docs.anaconda.com/anaconda/install/) to install QuantLab's prerequisites
```
$ cd ~/QuantLab
$ conda env create -f quantlab.yml
```
Note: at the moment of writing, the [PyTorch Anaconda channel](https://anaconda.org/pytorch/pytorch/files?version=) does not provide a distribution package for the combination Python 3.8/PyTorch 1.9 supporting a CUDA version lower than 10.2.
Therefore, if you plan to install QuantLab with GPU support, ensure that your CUDA version is not lower than 10.2.

After creating the Anaconda environment, it will make your life easier to install the `quantlib` quantization library in your Anaconda environment:
```
$ conda activate quantlab
$ (quantlab) cd quantlib
$ (quantlab) python setup.py build
$ (quantlab) python setup.py install
$ (quantlab) cd ..
```

### Configure storage drives
QuantLab was originally conceived to accelerate and organise experiments on workstations and (some) supercomputers.
We conceived a configuration mechanism to optimise execution on such systems, which must be set properly before using QuantLab.
You might find this part of the installation process slightly annoying if you have cloned and installed QuantLab to your personal computer.
We plan to make this step more transparent in the future, but for the moment there is no workaround: we are sorry.

An ML experiment takes as input a *machine learning system* and produces *logs* as outputs.
The machine learning system itself consists of three main parts:

* the **data set**;
* the hypothesis space (usually a space of parametric functions);
* the learning algorithm (in the case of deep neural networks this is an iterative application of both the backpropagation algorithm and a variant of stochastic gradient descent).

The **logs** include:

* checkpoints of the state of the system during the training run;
* statistics collected during the training run.

If you are used to reasoning in abstract terms, you'd probably never had to think that the access time to whatever resource you need is greater than zero, or that you can not store infinite amounts of information.
In practice, whereas the hypothesis space and the learning algorithm of an ML system occupy little more storage capacity than what is needed by their defining code, data and logs can take up gigabytes of storage space and quickly fill up the most capable storage devices.
Moreover, when training a machine learning model, you'd like to have high access **speed** to your data, since you don't want that the process which runs your program wastes time into waiting for data to be moved from/to storage to/from the computer's main memory.
Instead, you usually care less about spending a longer time storing the checkpoints and statistics about your training run: you will need them later for analysis, and you just care that whatever information you want to keep, you have sufficiently much **space** to store it.

Modern workstations usually include storage devices of two types that allow to trade access speed for storage capacity:
* hard disk drives (HDD), with very large capacity but slow access;
* solid state drives (SSD), which have smaller capacity than a typical HDD, but also provide much faster access.

Imagine that your machine has exactly one HDD and exactly one SSD.
Also, suppose that your operating system has a filesystem on which the devices are mounted to `/dev/sda1` and `/dev/sdb1`, respectively.
Finally, suppose that your filesystem exposes two folders:
* `/scratcha` mounted on `/dev/sda1` (i.e., physically being the HDD);
* `/scratchb` mounted on `/dev/sdb1` (i.e., physically being the SSD).

If you
```
$ (quantlab) ls QuantLab/configure
```
you will find a JSON file and three BASH scripts:
* `storage_cfg.json`;
* `storage.sh`;
* `problem.sh`;
* `topology.sh`.

To configure QuantLab, you need to tell it where the data for and logs of your experiments should be physically stored.
You do this by editing `storage_cfg.json`.
Continuing the example above, you would set
```
$ (quantlab) vim QuantLab/configure/storage_cfg.json
$ 
{
    'data': '/scratchb',
    'logs': '/scratcha'
}
:wq
$ 
```
so that the tool will fetch your data from the SSD drive and write logs to the HDD drive.
Running the `storage.sh` script creates *mock-up QuantLab homes* under both folders:
```
$ (quantlab) bash configure/storage.sh
$ (quantlab) ls /scratchb/QuantLab  # this folder now exists
$ (quantlab) ls /scratcha/QuantLab  # this folder now exists
```

QuantLab is shipped with example *problem packages* (`CIFAR10`, `ILSVRC12`), each of which contains one or more *topology sub-packages* (e.g., `VGG`).
When running an experiment in QuantLab, its abstractions will look for data in a `data` sub-folder under the chosen problem package, independently of the chosen network topology.
Instead, they will log results in a `logs` sub-folder under the chosen topology sub-package.
The `problem.sh` and `topology.sh` scripts actually create such folders on the devices specified at configuration time, and then create links to these folders under the problem and topology sub-package:
```
$ (quantlab) bash configure/problem.sh CIFAR10       # ~/QuantLab/systems/CIFAR10/data     -- now points to -> /scratchb/QuantLab/systems/CIFAR10/data
$ (quantlab) bash configure/topology.sh CIFAR10 VGG  # ~/QuantLab/systems/CIFAR10/VGG/logs -- now points to -> /scratcha/QuantLab/systems/CIFAR10/VGG/logs
```

### Run a QuantLab experiment
You can now run your first QuantLab experiment!
Configure the experiment
```
$ (quantlab) python main.py --problem=CIFAR10 --topology=VGG configure
$ [QuantLab] Experimental unit #0.
$ [QuantLab] Experimental unit's logs folder created at <~/QuantLab/systems/CIFAR10/VGG/logs/exp0000>.
```
and train the system
```
$ (quantlab) python main.py --problem=CIFAR10 --topology=VGG train --exp_id=0
$ [QuantLab] Experimental unit #0.
$ [QuantLab] No checkpoint found at <~/QuantLab/systems/CIFAR10/VGG/logs/exp0000/fold0/saves>.
$ ...
```


## Adding problems and topologies
Whenever you want to start working on a new data set, you should invoke the configuration script `problem.sh`.
Assuming that your data set is codenamed `XYZ`, issue the following commands:
```
$ (quantlab) cd ~/QuantLab
$ (quantlab) bash configure/problem.sh XYZ
$ [QuantLab] Remember to prepare the data for problem XYZ at </scratchb/QuantLab/systems/XYZ/data>."
$ (quantlab) ls ~/QuantLab/systems/XYZ  # this folder now exists
```
In particular, the warning `[QuantLab] Remember to prepare the data for problem XYZ at </scratchb/QuantLab/systems/XYZ/data>.` is raised to the user because QuantLab has no way to know how to prepare the files representing the data points of your data set.
In some cases, the raw-but-prepared data points might already be available on the computing systems where you installed QuantLab; for example, this is the case for many laboratory workstations that are shared amongst many people.
In such cases, assuming that the data set is stored in some folder `/scratchc/ml_datasets/XYZ`, we suggest the following workaround:
```
$ (quantlab) cd /scratchb/QuantLab/systems/XYZ
$ (quantlab) rm -r data
$ (quantlab) ln -s /scratchc/ml_datasets/XYZ data
```
In other words: you should manually replace the `data` folder automatically created by `problem.sh` inside `/scratchb/QuantLab/systems/XYZ` with a soft-link to the folder containing the raw-but-prepared data points.

To prepare the (empty) working files for a new network topology `ABC` that you want to apply to the problem `XYZ`, invoke the configuration script `topology.sh`:
```
$ (quantlab) cd ~/QuantLab
$ (quantlab) bash configure/topology.sh XYZ ABC
$ (quantlab) ls ~/QuantLab/systems/XYZ/ABC  # this folder now exists
```


## Accelerating training experiments with Horovod
Apart from CPU and GPU training, QuantLab supports multi-GPU training by wrapping PyTorch networks inside `nn.DataParallel` constructs.
If you need even more training speed, QuantLab also supports multi-process (and therefore potentially multi-node) training runs via Horovod.

Horovod is a software tool developed to simplify the execution of distributed training runs for deep learning models.
It is compatible with the most popular deep learning frameworks, including PyTorch and TensorFlow.
In case you want to benefit from this performance boost, you need to install Horovod according to the [official guide](https://horovod.readthedocs.io/en/stable/gpus_include.html).

Assuming that you installed Horovod inside the `quantlab` Anaconda environment, you can try the following to verify that everything works correctly:
```
$ (quantlab) cd ~/QuantLab
$ (quantlab) python main.py --problem=XYZ --topology=ABC configure
$ [QuantLab] Experimental unit #0.
$ [QuantLab] Experimental unit's logs folder created at <~/QuantLab/systems/CIFAR10/VGG/logs/exp0000>.
$ (quantlab) horovodrun -np 1 -H localhost:1 python main.py --problem=XYZ --topology=ABC train --exp_id=0
$ [QuantLab] Experimental unit #0.
$ [QuantLab] No checkpoint found at <~/QuantLab/systems/CIFAR10/VGG/logs/exp0000/fold0/saves>.
```
