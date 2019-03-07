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
- dlib: `pip install dlib`
- h5py: `pip install h5py`

Not all methods are used in all the libraries. Below are the instructions to install the requirements of each
specific method.

### OpenFace

- Follow instructions at https://cmusatyalab.github.io/openface/setup/

### FaceNet

- Clone repo https://github.com/davidsandberg/facenet

### VGG-Face

- Usar `pip install keras_vggface`

### face_recognition

- Usar `pip install face_recognition`

## Tutorials/Demos

## How to contribute

To contribute it is possible to perform a pull request. If your goal is to add a new method then
follow the next steps:

- Add a module in the [methods](ace_recognition_toolbox/methods) directory that contains a class with a `predict` method.
- Add the reference to the class [here](face_recognition_toolbox/methods/__init__.py)
- Add instance and call to the predict method in [utils.py](face_recognition_toolbox/utils.py)
