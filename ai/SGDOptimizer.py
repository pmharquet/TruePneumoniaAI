class SGDOptimizer:
    def __init__(self, layers, learning_rate=0.01, momentum=0.9):
        self.layers = layers
        self.lr = learning_rate
        self.momentum = momentum
        self._velocities = {}

    def step(self):
        for layer in self.layers:
            for param, grad in layer.get_params_and_grads():
                key = id(param)
                if key not in self._velocities:
                    # param * 0.0 crée un tableau nul du même type (numpy ou cupy)
                    # et du même device que le paramètre — pas besoin d'importer xp
                    self._velocities[key] = param * 0.0
                v = self._velocities[key]
                v *= self.momentum
                v += self.lr * grad
                param -= v

    def set_lr(self, new_lr):
        self.lr = new_lr
