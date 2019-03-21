"""
Module for managing the VGGFace recognition method.

Documentation can be found at https://github.com/rcmalli/keras-vggface
"""

import cv2
import numpy as np
from keras_vggface.vggface import VGGFace as VGGFaceModel


class VGGFace:
    def __init__(self, model='resnet50'):
        self.model = VGGFaceModel(model=model)

    def predict(self, image, normalize=True):
        """
        Get encoding of the face.

        Image will be resized to 224x224 using bicubic interpolation

        :param np.array image: Face image
        :param bool normalize: Return normalized vector
        :return: Face encoding
        """

        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        image = np.moveaxis(image, -1, 0)
        image = np.array([image], dtype=np.float64)

        rep = self.model.predict(image)

        if normalize:
            return rep.astype(np.float64) / np.linalg.norm(rep)
        else:
            return rep.astype(np.float64)
