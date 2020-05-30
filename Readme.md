# QuantLab

QuantLab is a tool to design, compare and select quantized neural networks (QNNs).
QuantLab has been developed to extend the PyTorch framework, and uses TensorBoard for its logging capabilities.

QuantLab consists of two main components:
* a **library** of quantization tools;
* **organising software** to manage machine learning (ML) experiments.

Before you proceed, be sure you digested this small caveat first.

**In its role of experiment management tool, QuantLab's software structure reflects my personal view of what a machine learning experiment should look like.**

Since it is likely that in your career you have faced (or you are facing) different needs from mine, it is very likely that you have different views on how to handle these complex processes.
If 1) you like spending time in formalising your program's designs and 2) you have fun discussing software engineering, I'd be happy to hear from you!


## Installation
In the following, I assume you have pulled the repository to your local machine at your home folder, which I will denote with `~`.

First, install Horovod's prerequisites by following the [official guide](https://horovod.readthedocs.io/en/stable/gpus_include.html).

Then, use [Anaconda](https://docs.anaconda.com/anaconda/install/linux/) to install QuantLab's prerequisites
```
$ cd QuantLab
$ conda env create -f quantlab.yml
$ cd ~
```


## Configuration
If you plan to use QuantLab as an experiment manager, grasping the structure of the project will help you.

I call the ```QuantLab``` folder the **QuantLab home**.
If you
```
$ ls QuantLab
```
you will see four major components:
* ```quantlab```,
* ```problems```,
* ```experiments```,
* ```main.py```.

The first three files are Python packages, whereas the fourth is the launch script for QuantLab experiments.

The ```quantlab``` package is the quantization library: its API exposes **graph editing functions** (```graphs```), **quantization operations** and **training controllers** (```algorithms```) and **statistics trackers** (```monitor```).
If you want to use QuantLab as a library in an existing project of yours, you can just copy the package in your project's folder (I'll have to make this an actual standalone Python package...).

Instead, the ```problems```, ```experiments``` and ```main.py``` components implement the experiments management functionality.
* Assumption #1.
  The typical problem a ML practitioner is asked to solve is to develop a ML model starting from a specific data set.
  Your lab's head want you to compare his new model A against a model B on a public data set like ImageNet or COCO; your boss wants you to analyse a proprietary CSV file about some market sales; your boss's boss urged your boss to force you into helping another team analysing their database about failures of mechanical components...
  You pick one.
* Assumption #2.
  Once you have the data set, you will probably need to try out different models: different data preprocessing routines, different network topologies, different configurations of training hyperparameters on the same topology...
* Assumption #3.
  At the end of your experiments batch, you would probably like to compare the statistics about your models in a straightforward way, possibly without spending a day looking through your filesystem for all the files which relate to the original data set.

To sum up: given a data set, first you would like to put it in a precise location, then describe a number of models to process and transform this data, pipe the statistics about the respective training runs to a second precise location, and finally inspect the files at this location to compare the outputs and therefore pick the best model.

This is what the ```problems``` package is for.
I call ```problem``` the **problems home**.
As a standard user, this is the only location you are supposed to access and modify in QuantLab.
In it, you can define multiple sub-packages called the **problem sub-packages**.
I conceived the problem sub-package abstraction as a container where you can perform exploratory data analysis on your data set, define the output data structures and metrics, describe DNN models, describe transformations of DNNs into QNNs, describe network-specific loss functions and pre-/post-processing routines, and finally specify training configurations.

As an example, I provide the ```ImageNet``` sub-package.
If you
```
$ ls QuantLab/problems/ImageNet
```
you will see the following components:
* ```meter.py```,
* ```ResNet```;

what is missing are the ```data``` and ```logs``` folders; I will explain why they are still not there in a moment.

The file ```meter.py``` essentially implements the concept of **metric** for a problem.
It does this by defining a ```Meter``` object, whose role is providing a standard, common set of evaluation functions whose outputs will allow to compare all the models designed to solve a specific problem.
For those of you who are familiar with the COCO data set, its role is similar to what scripts like ```cocoeval.py``` are meant to do.

The ```ResNet``` sub-sub-package is what I call a **library**.
A library implements all the topology-specific details of a DNN machine learning system: from the pre-processing functions, through the network definition and network-specific loss function, to the post-processing functions.
The role of a post-processing function is to transform the output of the network into a data structure which is compatible with what the ```Meter``` (and hence your application) expects.
For those of you familiar with object detection, this can be better understood through an example: all object detecion models should ultimately output a list of 5-uples describing bounding boxes and their content, but different networks require different post-processing to yield such a structure (e.g., YOLO uses *non-maximal suppression*).
These transforms are specific to the network, and I therefore think it is more natural to implement them inside the library.

To create the ```data``` and ```logs``` folder specific to your problem sub-package, the storage of your local machine has to be configured properly.
Why did I not put them directly into the repo?
A ML experiment has two inputs, the **data set** and the **model** (which includes the *program* and the *learning algorithm*), and one output, the **logs** (both training statistics and the program's parameters).
From an abstract perspective (e.g., if you are a mathematician), you'd probably never had to think that the access time to whatever resource you need is greater than zero.
Turns out, it does: whereas the model occupies just the space needed by its defining scripts, data and logs can take up gigabytes of storage space.
Moreover, different storage technologies respond to different physical needs.
When training a machine learning model, you'd like to have **fast** access to your data, since you don't want that the process which runs your program wastes time into waiting for data to arrive to the computer's main memory (the so-called *interrupts*).
Instead, you don't mind to spend a longer time storing the statistics that describe your training run, or the state of your model: you will need them later, and you just care that whatever information you want to keep, you have a sufficiently **large space** to store it.

In many modern computing systems, there are usually two types of storage, differentiated by technology:
* hard disk drives (HDD), with very large capacity but slow response time;
* solid state drives (SSD), which have smaller capacity than a typical HDD, but also much faster response time.

For simplicity, I suppose your machine has exactly one HDD and exactly one SSD.
I also suppose your operating system has a filesystem on which the devices are mounted to ```/dev/sda1``` and ```/dev/sdb1```, respectively.
Finally, I suppose the filesystem exposes two folders:
* ```/scratcha```, mounted on ```/dev/sda1``` (i.e., physically being the HDD);
* ```/scratchb```, mounted on ```/dev/sdb1``` (i.e., physically being the SSD).

If you
```
$ ls QuantLab/cfg
```
you will find three files:
* ```hard_folders.json```;
* ```storage.sh```;
* ```problem.sh```.

First, you need to tell QuantLab where your data and logs should be physically stored.
You do this by editing ```hard_folders.json```.
Continuing the example above, you would set:
```
{
    'data': '/scratchb',
    'logs': '/scratcha'
}
```

To simplify my life when navigating the filesystem of my machine, I decided to create two replicas of the problems home on the storage devices where your data and logs will live.
The ```storage.sh``` script accomplishes exactly this:
```
$ cd QuantLab/cfb
$ bash storage.sh
$ cd ~
# now folder /scratchb/QuantLab/problems exists
# now folder /scratcha/QuantLab/problems exists
# these are NOT the same folder as ~/QuantLab/problems
```

Note that no problem-specific data and log folders still exist.
Whenever you create a new problem, you should invoke the configuration script ```problem.sh```.
If I suppose your problem/data set is named ```XYZ```, invoking
```
$ cd QuantLab/cfg
$ bash problem.sh XYZ
$ cd ~
```
will
* create the "real" data folder at ```/scratchb/QuantLab/problems/XYZ/data```,
* create the "real" logs folder at ```/scratcha/QuantLab/problems/XYZ/logs```,
* create the ```XYZ``` problem sub-package at ```~/QuantLab/problems/XYZ```,
* create a symbolic link ```~/QuantLab/problems/XYZ/data``` to the "real" data folder,
* create a symbolic link ```~/QuantLab/problems/XYZ/logs``` to the "real" logs folder.
* create the file ```~/QuantLab/problems/XYZ/__init__.py``` (to ensure this is a Python package).

For example, after installing the repo and configuring the storage, you should run
```
$ cd QuantLab/cfg
$ bash problem.sh ImageNet
$ cd ~
```
to configure the data and logs folder for the ImageNet problem.
Of course, remember to install the actual data in the "real" data folder...


## Running
Once a problem package is in place, execution in QuantLab is straightforward.

To run the ImageNet example, invoke
```
$ cd QuantLab
$ horovodrun -np 1 -H localhost:1 python main.py --problem=ImageNet --topology=ResNet
```

Supposing your problem is named ```XYZ``` and your topology is called ```ABC```, just invoke
```
$ cd QuantLab
$ horovodrun -np 1 -H localhost:1 python main.py --problem=XYZ --topology=ABC
```