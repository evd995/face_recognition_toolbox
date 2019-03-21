"""
Module with useful functions for the package
"""

AVAILABLE_METHODS = [
    'FaceNet',
    'GoogleNet',
    'AlexNet',
    'SqueezeNet',
    'VGGFace',
    'OpenFace',
    'face_recognition'
]


def get_available_methods():
    print('Currently implemented methods:')
    print('\n'.join(AVAILABLE_METHODS))
    return AVAILABLE_METHODS


def predict(image, method_name, **kwargs):
    """
    Get descriptor of image with an specific method.
    """

    if method_name not in AVAILABLE_METHODS:
        raise NotImplementedError(
            "Method not currently supported. Implemented methods are:\n" + '\n'.join(AVAILABLE_METHODS))

    if method_name == 'FaceNet':
        from .methods import FaceNet
        model = FaceNet(**kwargs)
    elif method_name == 'GoogleNet':
        from .methods import GoogleNet
        model = GoogleNet(**kwargs)
    elif method_name == 'AlexNet':
        from .methods import AlexNet
        model = AlexNet(**kwargs)
    elif method_name == 'SqueezeNet':
        from .methods import SqueezeNet
        model = SqueezeNet()
    elif method_name == 'VGGFace':
        from .methods import VGGFace
        model = VGGFace(**kwargs)
    elif method_name == 'OpenFace':
        from .methods import OpenFace
        model = OpenFace(**kwargs)
    elif method_name == 'face_recogntion':
        from .methods import FaceRecognition
        model = FaceRecognition(**kwargs)

    return model.predict(image)
