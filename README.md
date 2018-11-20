# AI Programming with Python Project - Image Classification

## Description
#### Udacity description
Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.
#### My Description
A really fun project that brings together all the various components taught throughout the course into a decent challenge.
## Prerequisites
* Anaconda. You can download [here](https://www.anaconda.com/download/). This must be python 3.6 or higher. Or use whatever environment manager of your choice. Pytorch talks about `Chocolatey`. Never tried it, but it sounds yummy.
* `Pytorch` with `torchvision`. Follow the get started installation process as provided by pytorch.org [here](https://pytorch.org/get-started/locally/). I suggest taking python 3.7.
* Various dependencies such as `numpy` `pil` `matplotlib`

**Note:** Matplotlib with conda environment on windows...ok so this annoyed the life out of me for quite some time. Then eventually I ran into a very nice stackoverflow answer (can't remember the link now). But basically, in your conda environment, uninstall matplotlib, `conda update --all` and then re-install `matplotlib`. and BOOM it works.

## Data
They said don't commit to repo. So if you are reading this...I hope you have access to the data.

## Jupyter Notebook
Really simple.
1. Open up your anaconda prompt or whatever environment manager you use.
2. Start your environment.
3. Type `jupyter notebook`
4. A jupyter notebook will open in your browser
5. Click on `Image Classifier Project.ipynb`

## Command Line application
#### Train.py
Type `python train.py`

Here is a list of args that can be passed
* `--base_data_directory`
  * Default - `./flowers`
  * Type: `String`
  * Description: directory that contains the /train, /valid and /test data folders.
* `--arch`
  * Default - `densenet121`
  * Type: `String`
  * Choices:
    * densenet121
    * densenet169
    * vgg16
    * vgg11
    * alexnet
  * Description: Architectural model to use. Note pretained is always True.
* `--checkpoint`
  * Default - `./checkpoint.pth`
  * Type: `String`
  * Description: This is the directory to which the best trained model will be saved to.
* `--use_checkpoint`
  * Default - `False`
  * Type: `Boolean`
  * Description: Load checkpoint before training. This is useful when training in small epochs to test along the way.
* `--learning_rate`
  * Default - `0.001`
  * Type: `Integer`
  * Description: The rate at which the optimizer will train the model
* `--dropout`
  * Default - `0.5`
  * Type: `Integer`
  * Description: The classifier hidden layers dropoff rate
* `--training_iterations`
  * Default - `15`
  * Type: `Integer`,
  * Description: Number of epochs for the model to train.
* `--hidden_layer`
  * Default - `500`
  * Type: `Integer`
  * Description: Hidden layer size
* `--check_accuracy`
  * Default - `False`
  * Type: `Boolean`
  * Description: If True this will run through /test data and give a % accuracy on the data.
* `--device`
  * Default - `cuda`
  * Type: `String`
  * Description: If cuda is available leave this as default. Otherwise specify 'cpu'

#### Predict.py
Type `python predict.py`

Here is a list of args that can be passed
* `--image_dir`
  * Default - `./flowers/valid/18/image_04252.jpg`
  * Type: `String`
  * Description: This is the directory of the flower image you wish to predict on.
* `--image_category`
  * Default - `18`
  * Type: `String`
  * Description: This is the correct image category id for specified image
* `--topk`
  * Default - `10`
  * Type: `Integer`
  * Description: Number of top results/probabilities to display.
* `--checkpoint`
  * Default - `./checkpoint.pth`
  * Type: `String`
  * Description: This is the directory to the prediction model.
