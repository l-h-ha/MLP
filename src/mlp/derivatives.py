import numpy as np
from mlp.base_object import base_object

class IDENTITY(base_object):
    def __init__(self):
        self.obj_type = "derivative"
    
    def __call__(self, x: np.ndarray):
        return 0

class RELU(base_object):
    def __init__(self):
        self.obj_type = "derivative"
    
    def __call__(self, x: np.ndarray):
        return np.where(x > 0, 1, 0)

class LEAKY_RELU(base_object):
    def __init__(self, alpha: float = 0.1):
        self.obj_type = "derivative"
        self.alpha = alpha
    
    def __call__(self, x: np.ndarray):
        return np.where(x > 0, 1, self.alpha)
    
class SOFTMAX(base_object):
    def __init__(self):
        self.obj_type = "derivative"
    
    def __call__(self, x: np.ndarray):
        return x * (1 - x)
    
class STABLE_SOFTMAX(base_object):
    def __init__(self):
        self.obj_type = "derivative"
    
    def __call__(self, x: np.ndarray):
        return x * (1 - x)