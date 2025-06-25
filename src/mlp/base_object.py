class base_object():
    def __init__(self, obj_type: str):
        self.obj_type = obj_type
    
    def get_type(self) -> str:
        return self.obj_type

    def is_type(self, obj_type: str) -> bool:
        return self.obj_type == obj_type