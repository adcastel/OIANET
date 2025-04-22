import time
class BaseModel:
    def __init__(self, layers):
        self.layers = layers
    
    def get_model(self):
        return self.layers

    def forward(self, x, curr_iter=1):
        imgs=x.shape[0]
        if curr_iter == 0:
            print("FW Layer;Batch;Time(s);Performance(imgs/s)")
        for layer in self.layers:
            layer_start_time = time.time()
            x = layer.forward(x)
            layer_time = time.time() - layer_start_time
            if curr_iter == 0:
                # Calculate performance metrics
                images_per_second = imgs / layer_time
                print(f"{layer.__class__.__name__};{imgs};{layer_time:.4f};{images_per_second:.2f}")
        if curr_iter == 0:
            print("==========================================")
        
        return x

    def backward(self, grad_output, learning_rate,curr_iter=1):
        imgs=len(grad_output)
        if curr_iter == 0:
            print("BW Layer;Batch;Time(s);Performance(imgs/s)")
        for layer in reversed(self.layers):
            layer_start_time = time.time()
            grad_output = layer.backward(grad_output, learning_rate)
            layer_time = time.time() - layer_start_time
            if curr_iter == 0:
                images_per_second =  imgs/ layer_time
                print(f"{layer.__class__.__name__};{imgs};{layer_time:.4f};{images_per_second:.2f}")
        if curr_iter == 0:
            print("==========================================")
        return grad_output
