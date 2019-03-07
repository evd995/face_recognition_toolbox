import face_recognition
import numpy as np
import dlib


class FaceRecognition:
    def __init__(self):
        pass

    def predict(self, image, normalize=True):

        bb = (0, image.shape[1], image.shape[0], 0)
        encoding = face_recognition.face_encodings(
            image, known_face_locations=[bb])

        if normalize:
            return encoding[0].astype(np.float64) / np.linalg.norm(encoding[0])
        else:
            return encoding[0].astype(np.float64)
