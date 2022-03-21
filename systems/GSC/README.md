# Google Speech Commands
The [Google Speech Commands (GSC) data set](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) is a popular speech recognition data set.
GSC encodes several keyword spotting tasks.
A keyword spotting system must correctly classify word utterances, mapping user commands to the corresponding entries in a given vocabulary.
Therefore, accurate keyword spotting systems can greatly facilitate the interaction of users with smart devices.

In QuantLab, we use the [GSCv2](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz) version of the data set.
It contains 105830 recordings of 35 different words uttered by 2619 different speakers, plus five recordings of different background noise types.


## Installing the GSC data set
Assuming that you have already created the QuantLab problem package `GSC` as detailed in the main README, you can install the GSC data set by issuing the following commands:
```
(quantlab) $ cd systems/GSC/data/
(quantlab) $ wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
(quantlab) $ tar xzf speech_commands_v0.02.tar.gz
(quantlab) $ cd ../../../
```
