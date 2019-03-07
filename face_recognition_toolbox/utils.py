def predict(image, method_name, **kwargs):
    if method_name == 'FaceNet':
        from .methods import FaceNet
        model = FaceNet(**kwargs)
    elif method_name == 'GoogleNet':
        from .methods import GoogleNet
        model = GoogleNet(**kwargs)
    elif method_name == 'AlexNet':
        from .methods import AlexNet
        model = AlexNet()
    elif method_name == 'SqueezeNet':
        from .methods import SqueezeNet
        model = SqueezeNet()
    elif method_name == 'VGGFace':
        from .methods import VGGFace
        model = VGGFace()
    elif method_name == 'OpenFace':
        from .methods import OpenFace
        model = OpenFace(**kwargs)
    elif method_name == 'face_recogntion':
        from .methods import FaceRecognition
        model = FaceRecognition()

    return model.predict(image)
