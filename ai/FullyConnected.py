import numpy as np
from Neuron import Neuron


class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = [
            Neuron(
                weights=np.random.randn(input_size) * np.sqrt(2.0 / input_size),
                bias=0.0
            )
            for _ in range(output_size)
        ]

    def forward(self, input_data):
        return np.array([neuron.forward(input_data) for neuron in self.neurons])

    def backward(self, grad):
        d_input = np.zeros(self.input_size)
        for i, neuron in enumerate(self.neurons):
            neuron.backward(grad[i])
            d_input += grad[i] * neuron.weights
        return d_input

    def zero_grads(self):
        for neuron in self.neurons:
            neuron.d_weights[:] = 0.0
            neuron.d_bias[0] = 0.0

    def get_params_and_grads(self):
        result = []
        for neuron in self.neurons:
            result.extend(neuron.get_params_and_grads())
        return result
