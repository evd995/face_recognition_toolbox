"""
Module for managing the FaceNet recognition method.

2018-04-08:
    FaceNet has two pretrained models available, one trained
    with the CASIA-WebFace dataset and the other with VGGFace2.
    Both are trained with an Inception ResNet v1 arquitecture.

    The name of the respective models are:
        - facenet-20180408-102900.pb (for CASIA-WebFace)
        - facenet-20180402-114759.pb (for VGGFace2, default)

    Pre-trained models can be downloaded at https://github.com/davidsandberg/facenet#pre-trained-models
"""

import tensorflow as tf
import numpy as np
import cv2
import os

IMAGE_SIZE = 160


class FaceNet:

    def __init__(self, model='facenet-20180402-114759.pb'):
        """
        Loads a FaceNet instance with the specified model (it has to exist in "weights" directory).

        This method loads a frozen graph because it is less memory 
        consuming than loading the whole model.

        :param str model: Name of the file of the model (it has to exist in "weights" directory).
        """
        print('Load Frozen Graph')

        with tf.gfile.FastGFile(os.path.join(os.path.dirname(__file__), "weights", model),
                                'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
            self.graph = tf.get_default_graph()

        print('Ended loading frozen graph')

    def predict(self, image, normalize=True):
        """
        Get encoding of the face.

        Image will be resized to 160x160 using bicubic interpolation

        :param np.array image: Face image
        :param bool normalize: Return normalized vector
        :return: Face encoding
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
        
        # Facenet prewhiten
        mean = np.mean(image)
        std = np.std(image)
        std_adj = np.maximum(std, 1.0/np.sqrt(image.size))
        image = np.multiply(np.subtract(image, mean), 1/std_adj)
        
        images = [image]

        with tf.Session(graph=self.graph) as sess:
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = {images_placeholder: images,
                         phase_train_placeholder: False}
            emb = sess.run(embeddings, feed_dict=feed_dict)

            if normalize:
                return emb[0, :].astype(np.float64) / np.linalg.norm(emb[0, :])

            else:
                return emb[0, :].astype(np.float64)
