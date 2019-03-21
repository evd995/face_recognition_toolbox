"""
WARNING: OpenFace requires Python 2.7
Module for managing the OpenFace recognition method.

To install this method follow instructions at https://cmusatyalab.github.io/openface/setup/
"""

import time
import os
import numpy as np
import cv2
import sys


def getRGB(image_path):
    bgr_img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb_img


class OpenFace:
    def __init__(self, fileDir, model='nn4.small2.v1.t7'):
        """
        Creates instance of OpenFace model.

        Requires as input:
            - fileDir: path of OpenFace Directory (of git clone)
            - model: (optional) file name of the model
        """

        if sys.version_info[0] != 2:
            raise ImportError("OpenFace requires Python 2.7")
        else:
            import openface

        # Get paths of dlib model and predictors
        # OpenFace Directory (of git clone)
        modelDir = os.path.join(fileDir, 'models')
        openfaceModelDir = os.path.join(modelDir, 'openface')

        # Parameters for AlignDlib (passed through console in example)

        networkModel = os.path.join(openfaceModelDir, model)
        self.imgDim = 96

        self.model = openface.TorchNeuralNet(networkModel, self.imgDim)

    def predict(self, image, normalize=True):
        """
        Get encoding of the face.

        Image will be resized to 224x224 using bicubic interpolation

        :param np.array image: BGR face image
        :param bool normalize: Return normalized vector
        :return: Face encoding
        """

        rgbImg = getRGB(image)
        rep = self.model.forward(cv2.resize(rgbImg, (self.imgDim, self.imgDim),
                                            interpolation=cv2.INTER_CUBIC))

        if normalize:
            return rep.astype(np.float64) / np.linalg.norm(rep)
        else:
            return rep.astype(np.float64)
