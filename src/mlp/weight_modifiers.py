import numpy as np
from mlp.base_object import base_object

class HE_init(base_object):
    def __init__(self):
        self.obj_type = "weight_modifier"

    def init(self, this_shape, next_shape):
        return np.random.rand(this_shape[1], next_shape[1]) * np.sqrt(2 / this_shape[1])