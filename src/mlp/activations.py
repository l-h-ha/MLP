import numpy as np
from mlp.base_object import base_object

class identity(base_object):
    def __init__(self):
        self.obj_type = "activation"
    
    def __call__(self, x: np.ndarray):
        return x

class relu(base_object):
    def __init__(self):
        self.obj_type = "activation"

    def __call__(self, x: np.ndarray):
        return np.maximum(0, x)
    
class leaky_relu(base_object):
    def __init__(self, alpha: float = 0.1):
        self.obj_type = "activation"
        self.alpha = alpha

    def __call__(self, x: np.ndarray):
        return np.maximum(self.alpha * x, x)
    
class softmax(base_object):
    def __init__(self):
        self.obj_type = "activation"
    
    def __call__(self, x: np.ndarray):
        e_x = np.exp(x)
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
class stable_softmax(base_object):
    def __init__(self):
        self.obj_type = "activation"
    
    def __call__(self, x: np.ndarray):
        e_x = np.exp(x - np.sum(e_x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)