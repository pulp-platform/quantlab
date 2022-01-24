# DVS128 in QuantLab

This folder contains a QuantLab Problem for the [DVS128 Dataset by
IBM](https://www.research.ibm.com/dvsgesture/). We provide a single
topology, `dvs_cnn` to solve the gesture classification problem using a
fully ternarized CNN+TCN hybrid architecture.

## Paper

Download the paper here: [ETHZ Research
Collection](https://www.research-collection.ethz.ch/handle/20.500.11850/527816)

## Setup

To run the experiments, you first need to run the `make_dataset.py`
script from this folder:

1.  The data conversion script uses OpenCV, which is not compatible with
    Python 3.8, so you will need to use a different virtual environment.
    The `opencv.yml` contains a
    [Conda](https://docs.conda.io/en/latest/miniconda.html) environment
    specification containing all required packages.
2.  Unpack the dataset to `my/data/folder`
3.  Create a symlink called `dvs128` to the dataset from this folder:
    `ln -s my/data/folder ./dvs128`
4.  Run the script - it will create about 50GB of data in a subfolder of
    the dataset, i.e., `my/data/folder/processed`: `python
    ./make_dataset.py` This will take a while and create a bunch of
    NumPy files containing the raw frames extracted from the event data.
5.  Link the `data` folder in your "hard storage" directory to the
    `processed` dataset folder.

## Pretrained Checkpoints:

You can find full precision checkpoints for all supported downsampling
configs (1x, 2x, 4x) and number of channels (32, 64, 96, 128) here:
<https://iis-people.ee.ethz.ch/~datasets/DVS_TCN/full_precision_checkpoints.tar.gz>
The full experiment log directory of the experiment used for the paper
results can be found here:
<https://iis-people.ee.ethz.ch/~datasets/DVS_TCN/paper_network.tar.gz>
You can copy the directory in this file into your `logs` folder and look
at the training curves in tensorboard.

## Network

The DVS CNN network is a hybrid ternarized neural network consisting of
a 2D CNN and a 1D TCN. The 2D CNN takes multiple ternary event frames as
input and outputs a single ternary 1D feature vector. The TCN then
processes multiple of these feature vectors and outputs a
classification/sequence of classifications (depending on the
`classifier_type` flag). The INQ quantization methods support the
following options, to be set in
`config.json`:

| Flag                                  | Data Type | Effect                                                                                                                                                                                                                                                                          |
| ------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DATASET FLAGS                         |           |                                                                                                                                                                                                                                                                                 |
| `transform/kwargs/downsample`         | integer   | Input images will be downsampled by this factor **in each direction**, i.e., a value of 2 will result in 64x64 inputs                                                                                                                                                           |
| `transform/kwargs/flip`               | bool      | Probability of flipping an input pixel's polarity                                                                                                                                                                                                                               |
| `load_data_set/kwargs/window_stride`  | integer   | Dataset generation: The windows containing all input frames that are fed into the network for a single classification will be spaced by this many frames. Lower numbers thus mean more batches per epoch.                                                                       |
| `load_data_set/kwargs/cnn_stride`     | integer   | The windows fed into the CNN to generate a single 1D feature vector will be spaced by this many frames. Lower numbers thus mean more overlap between the windows and as such a shorter receptive time window.                                                                   |
| `load_data_set/kwargs/cnn_win`        | integer   | Number of frames fed into the CNN, i.e., number of input channels of the first layer                                                                                                                                                                                            |
| `load_data_set/kwargs/tcn_win`        | integer   | Input window length to the TCN. Higher number -\> longer receptive time window                                                                                                                                                                                                  |
| `load_data_set/kwargs/n_val_subjects` | integer   | Number of validation subjects. Keep at 6 to reproduce paper results                                                                                                                                                                                                             |
| NETWORK FLAGS                         |           |                                                                                                                                                                                                                                                                                 |
| `kwargs/{tnn/cnn}_cfg_key`            | string    | Network configuration. Supported: `{32/64/96/128}_channels`                                                                                                                                                                                                                     |
| `kwargs/pool_type`                    | string    | Type of pooling used in the CNN. Supported: `max_pool`, `avg_pool`, `stride`. ~stride means no pooling layers are used, instead CNN conv layers have a stride of 2                                                                                                              |
| `kwargs/cnn_window`                   | integer   | Same as in dataset.                                                                                                                                                                                                                                                             |
| `kwargs/tcn_window`                   | integer   | Same as in dataset.                                                                                                                                                                                                                                                             |
| `kwargs/n_classes`                    | integer   | Must be 11                                                                                                                                                                                                                                                                      |
| `kwargs/classifier_bias`              | bool      | Whether to allow for a bias in the classifier. For a fully ternarized network/CUTIE mappability, leave false.                                                                                                                                                                   |
| `kwargs/twn_classifier`               | bool      | Whether to not quantize the inputs to the classifier layer. For a fully ternarized network/CUTIE mappability, leave false.                                                                                                                                                      |
| `kwargs/k_cnn`                        | integer   | Kernel size of CNN.                                                                                                                                                                                                                                                             |
| `kwargs/classifier_type`              | string    | Classifier: can be `learned` (classifier is a 1x1 convolution and will produce a sequence of length `tcn_window`) or `linear` (classifier is a `tcn_window` x 1 convolution and produces a single vector of class score)                                                        |
| `kwargs/classifier_out`               | string    | How to use the output of a `learned` classifier: If this flag is `all`, the entire sequence will be returned. If this flag is `last`, only the last element of the sequence will be returned                                                                                    |
| `kwargs/layer_order`                  | string    | Order of BN and pooling layers. Can be `bn_pool` or `pool_bn`; **must be `bn_pool`** to allow for correct ternarization\!                                                                                                                                                       |
| `kwargs/pretrained`                   | string    | Path to pretrained unquantized checkpoint (see below for links to pretrained nets)                                                                                                                                                                                              |
| `kwargs/fix_cnn_pool`                 | bool      | Make last layer's pooling CUTIE-compatible (include a 1x1 convolution). Leave true for CUTIE-mappability.                                                                                                                                                                       |
| `kwargs/last_conv_nopad`              | bool      | To reduce the number of layers and make the network CUTIE compatible, the last conv layer has no padding. This is needed when not downsampling at all or downsampling by 2x. When downsampling by 4x, this must be set to false as the feature map will be too small otherwise. |

Other flags should be self-explanatory.
