# Google Speech Commands
The [Google Speech Commands (GSC) data set](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) is a data set containing...
In our experiments, we use the extended [GSCv2](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz).

Assuming that you have already created the QuantLab problem package `GSC` as detailed in the main README, you can install the GSC data set by issuing the following commands:
```
(quantlab) $ cd systems/GSC/data/
(quantlab) $ wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
(quantlab) $ tar xzf speech_commands_v0.02.tar.gz
(quantlab) $ cd ../../../
```
