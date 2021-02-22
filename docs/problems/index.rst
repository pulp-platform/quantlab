``problems``
============


Needs and tools
---------------

The evolution of technology shifts the horizon of needs always further. For
example, the possibility of taking pictures cheaply using smartphones and
sharing them in real-time on social networks creates the need to remain
up-to-date with the most recent activities of your acquaintances (hopefully
avoiding to spend your whole day scrolling through all their photos); again,
the possibility of registering the movements of your hand (e.g., using a
small radar) creates the need to control devices using gestures; another
example: the possibility to register sounds creates the need to instruct
devices to solve tasks by issuing vocal commands.

Satisfying a new need requires developing some *ad-hoc* tool. The development
of a complex tool can be effectively faced by decomposing it in a pipeline of
simpler tools, each of which should solve a well-defined **task**. Some tasks
are currently too complex (or not sufficiently understood) to derive
analytical solutions with closed-form specifications that can be translated
into algorithms. In these cases, machine learning comes to the rescue. Each
machine learning task can belong to one or multiple **domains**:

* image classification, image segmentation, object detection, pose estimation
  belong to the domain of computer vision (CV);
* sentiment analysis, document classification, language understanding,
  language translation belong to the domain of natural language processing
  (NLP);
* speech recognition and MI-BCI (motion imagery, brain-computer interface)
  belong to the domain of biological signal processing.

Now that we have defined this taxonomy in terms of domains and tasks, we
conclude this introductory section with some examples of the *divide et
impera* tool development methodology introduced above. The need to remain
up-to-date with your acquaintances' activities can be satisfied using a
query-based tool, where the user would be asked to select a set of
"interesting" topics (a "filter") and would receive a selection of relevant
pictures found amongst those of his contacts. Then, the tool should 1) assign
each picture a label and 2) search in the set of all labelled images for those
whose label is in the set of the "interesting" topics selected by the user;
the first task can be classified as an image classification (machine learning)
task, whereas the second task is a classical (algorithmic) search in a list.
The need to instruct devices using vocal commands can instead be satisfied by
a tool that 1) translates the sound signals collected by a microphone into a
string of natural language and 2) understand the intention of the user, then
translate this into a formal query which can be executed by the device; the
first task is a speech recognition task, whereas the second is a language
understanding task.

.. warning:: Formalise these examples mathematically!


A zoo of problems and models
----------------------------

Machine learning tasks are actually classes grouping multiple problems. Each
problem can be functionally framed as a task instance, but the specific format
and metrics of the data which need to be processed, together with the
structural constraints these impose on the class of models that can be used,
hamper the portability of the solution (i.e., the actual computer program that
solves the problem) to other problems belonging to the same task. For example,
an image classification model designed to solve the MNIST data set will likely
not work on RGB images.

According to this definition, QuantLab includes a ``problems`` package to
structure exploratory data analysis functions, model definition and metric
definitions related to different data sets into separated (but coherent)
units. Each of these units is called a **problem sub-package**, and is
structured as follows:

* ``data`` folder: it contains (the pointers to) the folders containing the
  data set points; if you run exploratory data analysis on your data set, this
  is the place where the functions should be stored;

.. warning:: Create a ``data_exploration`` sub-folder? Should this be made a
   Python module? Then also ``data`` should become a module?

* ``meter.py`` module: it defines the :py:object:`Meter` object implementing
  the data set-specific metrics and comparing the output of a network with the
  label provided by the data set (in the case of supervised learning tasks);
* **topology sub-package(s)**: each of these Python packages contains multiple
  modules, including network topology description, graph quantization recipes,
  pre- and post-processing functions (in particular, the post-processing
  functions should convert the output of the network to a data structure which
  is compatible with what the problem's :py:object:`Meter` expects) and the
  (editable) experiment configuration file (``config.json``);
* ``logs`` folder: it contains (the pointers to) the folders containing the
  results of the experiments.

.. warning:: Document the "default" problem package (``ImageNet``) and its
   main library subpackages (``VGG``, ``ResNet``).
