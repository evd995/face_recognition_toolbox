"""
WARNING: OpenFace requires Python 2.7
Module for managing the GoogleNet recognition method.

Obtained from https://github.com/kgrm/face-recog-eval
"""

import os
import cv2
import numpy as np
from keras import backend as K
from .networks_def import InceptionV3


class GoogleNet:
    def __init__(self):
        self.model = InceptionV3(include_top=False)

        model_path = os.path.join(os.path.dirname(
            __file__), "weights", "googlenet_weights.h5")
        self.model.load_weights(model_path, by_name=True)
        print('--- Weights loaded ---')

        K.set_image_dim_ordering('th')

    def predict(self, image, normalize=True):
        """
        Get encoding of the face.

        Image will be resized to 299x299 using bicubic interpolation

        :param np.array image: Face image
        :param bool normalize: Return normalized vector
        :return: Face encoding
        """

        # Image preprocessing
        image = cv2.resize(image, (299, 299), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Array preprocesing
        image = np.moveaxis(image, -1, 0)
        image = np.array([image], dtype=np.float64)

        rep = self.model.predict(image)

        if normalize:
            return rep.astype(np.float64) / np.linalg.norm(rep)

        else:
            return rep.astype(np.float64)
