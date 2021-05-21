.. _systems-package:

``systems``
============


Needs and tools
---------------

The evolution of technology shifts the horizon of needs always further.
For example, the possibility of taking pictures cheaply using smartphones and sharing them in real-time on social networks creates the need to remain up-to-date with the most recent activities of your acquaintances (hopefully avoiding to spend your whole day scrolling through all their photos).
Again, the possibility of registering the movements of your hand (e.g., using a small radar) creates the need to control devices using gestures.
Another example: the possibility to register and classify sounds creates the need to instruct devices to solve tasks by issuing vocal commands.

Satisfying a new need requires developing some *ad-hoc* tool.
The development of a complex tool can be effectively faced by decomposing it using a *divide et impera* approach.
According to this approach, a complex tool is designed as a pipeline of simpler tools, each of which should solve a well-defined *task*.
Some tasks are currently too complex (or not sufficiently understood) to derive analytical solutions with closed-form specifications that can be translated into algorithms.
In these cases, machine learning systems might to the rescue.

Each task can belong to one or multiple *domains*:

* image classification, image segmentation, object detection, pose estimation
  belong to the domain of computer vision (CV);
* sentiment analysis, document classification, language understanding,
  language translation belong to the domain of natural language processing
  (NLP);
* speech recognition and MI-BCI (motion imagery, brain-computer interface)
  belong to the domain of biological signal processing.

Now that we have defined this taxonomy in terms of domains and tasks, we can provide some examples of the *divide et impera* tool development methodology introduced above.

The need to remain up-to-date with your acquaintances' activities can be satisfied using a query-based tool arranged according to a client-server pattern.
When the user installs the client on his device, he will be asked to select a set of "interesting" topics (i.e., to define a filter) and would daily receive a selection of relevant pictures found amongst those of his contacts.
In exchange, he will send the pictures he takes and uploads to the application to the server.
The tool should:

1. on the server side, assign each picture of its users a label;
2. on the client side, query the server to retrieve the set of all images belonging to acquaintances of the user and whose labels appear in the set of the "interesting" topics selected by the user.

The first task can be classified as an image classification task (that can be solved using a machine learning system), whereas the second task is a simple filtering operation over a list (that can be solved using standard algorithms).

The need to instruct devices using vocal commands can instead be satisfied by a tool performing the following tasks:

1. translate the sound signals collected by a microphone into a UNICODE string of natural language;
2. tokenise this string into a sequence of abstract actions (each of which can be performed using a simple algorithm);
3. execute each action.

The first task can be classified as a speech recognition task, the second as a language understanding task, and the last as the traversal of a list.
The first two tasks can be solve using machine learning systems, whereas standard algorithms should suffice to carry out the last one.


A zoo of problems and topologies
--------------------------------

From the definitions above, it should be clear that tasks are actually classes grouping multiple problems.
Each problem can be functionally framed as a task instance, but the specific format of the data points and the problem metrics, together with the structural constraints that they impose on the class of models that can be used, hamper the portability of the solution (i.e., the actual computer program that solves the problem) to other problems belonging to the same task class.
For example, we can not assume that an image classification model designed to solve the MNIST data set will also work on RGB images.

In the context of *deep learning*, solving a task on two different data sets usually requires re-designing the DNN topology, the pre-processing and the post-processing too.
For this reason, QuantLab arranges...


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   workingfiles
