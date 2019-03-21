# Face Recognition Module

This repository contains tools and different method to peform facial recognition. The algorithms
that are supported so far are:

- OpenFace: https://cmusatyalab.github.io/openface/
- FaceNet: https://github.com/davidsandberg/facenet
- VGG-Face: https://github.com/rcmalli/keras-vggface
- face_recognition (dlib): https://github.com/ageitgey/face_recognition y https://github.com/davisking/dlib-models
  (68 point shape predictor)
- AlexNet: https://github.com/kgrm/face-recog-eval
- GoogleNet (Inception-V3): https://github.com/kgrm/face-recog-eval
- SqueezeNet: https://github.com/kgrm/face-recog-eval
- SphereFace: https://github.com/clcarwin/sphereface_pytorch

The main goal of this toolbox is to compare the performance of different facial recognition methods
as done in [this paper](http://dmery.sitios.ing.uc.cl/Prints/Conferences/International/2019-WACV.pdf)

## Setup

Libraries used:

- Numpy: `pip install numpy`
- Matplotlib: `pip install matplotlib`
- OpenCV: `pip install opencv-python`
- Tensorflow: `pip install tensorflow`
- Keras: `pip install keras`
  - AlexNet, GoogleNet and SqueezeNet can only run in Keras 1.2
- dlib: `pip install dlib`
- h5py: `pip install h5py`

Not all methods are used in all the libraries. Below are the instructions to install the requirements of each
specific method.

### OpenFace

`WARNING: OpenFace can only run in Python 2.7`

- Follow instructions at https://cmusatyalab.github.io/openface/setup/

### FaceNet

- Download pre-trained models at https://github.com/davidsandberg/facenet#pre-trained-models
- Move weight to the [weight directory](face_recognition_toolbox/methods/weights).
- When running `predict` it is necessary to specify the `model` parameter as the name of the file of the pre-trained model

- 2018-04-08:
  FaceNet has two pretrained models available, one trained
  with the CASIA-WebFace dataset and the other with VGGFace2.
  Both are trained with an Inception ResNet v1 arquitecture.
  The name of the respective models are:

  - facenet-20180408-102900.pb (for CASIA-WebFace)
  - facenet-20180402-114759.pb (for VGGFace2, default)

  Pre-trained models can be downloaded at https://github.com/davidsandberg/facenet#pre-trained-models

### VGG-Face

- Run `pip install keras_vggface`
- 2019-03-21: As of today, `keras_vggface` supports the models: vgg16, resnet50 and senet50. The default is
  resnet50 but it can be specified by passing the `model` parameter with the desired model name.

### face_recognition

- Run `pip install face_recognition`

### AlexNet, GoogleNet and SphereFace

`WARNING: AlexNet, GoogleNet and SphereFace can only run in Python 2.7 and Keras 1.2.2`

- Download weights at https://github.com/kgrm/face-recog-eval
- Move weight to the [weight directory](face_recognition_toolbox/methods/weights).
- If currently installed version of Keras is greater than 1.2.2, it has to be downgraded before running this method with the command `pip install --upgrade keras==1.2.2`

## Tutorials/Demos

- A simple impletentation can be found in [this notebook](tests/Run_methods.ipynb)

## How to contribute

To contribute it is possible to perform a pull request. If your goal is to add a new method then
follow the next steps:

- Add a module in the [methods](face_recognition_toolbox/methods) directory that contains a class with a `predict` method.
- Add the reference to the class [here](face_recognition_toolbox/methods/__init__.py)
- Add instance and call to the predict method in [utils.py](face_recognition_toolbox/utils.py)
