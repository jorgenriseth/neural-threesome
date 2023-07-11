from dolfin import UserExpression

class Characteristic(UserExpression):
    def __init__(self, tags, value, **kwargs):
        self.tags = tags
        self.value = value
        super().__init__(**kwargs)
        
    def eval_cell(self, values, x, ufl_cell):
        if self.tags[ufl_cell.index] == self.value:
            values[0] = 1
        else:
            values[0] = 0
        
    def value_shape(self):
        return ()