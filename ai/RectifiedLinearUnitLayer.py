# Pas d'import numpy/cupy — les opérateurs element-wise fonctionnent
# de façon identique sur les tableaux NumPy et CuPy.

class RectifiedLinearUnitLayer:
    def __init__(self):
        self._last_input = None

    def forward(self, input_data):
        self._last_input = input_data
        return input_data * (input_data > 0)

    def backward(self, grad):
        return grad * (self._last_input > 0)
