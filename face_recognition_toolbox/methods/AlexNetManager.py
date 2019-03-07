import os
import cv2
import numpy as np
from .networks_def import AlexNet as AlexNetDef
from keras import backend as K


class AlexNet:
    def __init__(self):
        self.model = AlexNetDef(N_classes=1000, r=1e-4, p_dropout=0.5,  borders="same",
                                inshape=(3, 224, 224), include_softmax=False)

        model_path = os.path.join(os.path.dirname(
            __file__), "weights", "alexnet_weights.h5")
        self.model.load_weights(model_path, by_name=True)
        print('--- Weights loaded ---')

        K.set_image_dim_ordering('th')

    def predict(self, image, normalize=True):
        """
        Images are resized to 224x224
        """
        # Image preprocessing
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Array preprocesing
        image = np.moveaxis(image, -1, 0)
        image = np.array([image], dtype=np.float64)

        rep = self.model.predict(image)

        if normalize:
            return rep.astype(np.float64) / np.linalg.norm(rep)

        else:
            return rep.astype(np.float64)
