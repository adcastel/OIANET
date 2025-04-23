import numpy as np
import os
class Layer:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad_output, learning_rate):
        raise NotImplementedError
    
    def save_weights(self, path):
        os.makedirs(path, exist_ok=True)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_weights'):
                np.savez(os.path.join(path, f'layer_{i}.npz'), **layer.get_weights())

    def load_weights(self, path):
        for i, layer in enumerate(self.layers):
            weight_path = os.path.join(path, f'layer_{i}.npz')
            if hasattr(layer, 'set_weights') and os.path.exists(weight_path):
                data = np.load(weight_path)
                layer.set_weights({k: data[k] for k in data.files})