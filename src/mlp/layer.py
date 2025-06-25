from mlp.base_object import base_object

class layer(base_object):
    def __init__(self, shape: tuple):
        self.w = None
        self.b = None

        self.z = None
        self.a = None

        self.shape = shape
        self.obj_type = "layer"