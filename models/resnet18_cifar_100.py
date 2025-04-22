from modules.conv2d import Conv2D, Conv2D_np
from modules.relu import ReLU, ReLU_np
from modules.maxpool2d import MaxPool2D, MaxPool2D_np
from modules.flatten import Flatten, Flatten_np
from modules.dense import Dense, Dense_np
from modules.softmax import Softmax, Softmax_np
from modules.avgpool2d import GlobalAvgPool2D
from modules.layer import Layer
import time
class BasicBlock:
    def __init__(self, in_channels, out_channels, stride=1, use_im2col=False):
    
        self.use_projection = (in_channels != out_channels) or (stride != 1)
        self.stride = stride

        self.conv1 = Conv2D_np(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, use_im2col=use_im2col)
        self.relu1 = ReLU_np()
        self.conv2 = Conv2D_np(out_channels, out_channels, kernel_size=3, stride=1, padding=1, use_im2col=use_im2col)
        self.relu2 = ReLU_np()

        if self.use_projection:
            self.projection = Conv2D_np(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, use_im2col=use_im2col)
        else:
            self.projection = None
        self.first=True
    
    def forward(self, x):
        self.input = x
        imgs = x.shape[0]

        layer_start_time = time.time()  # Start timer for the layer
        out = self.conv1.forward(x)
        layer_time = time.time() - layer_start_time
        images_per_second = imgs / layer_time
        if self.first:
            print(f"Layer: {self.conv1.__class__.__name__}, Time: {layer_time:.4f}s, Performance: {images_per_second:.2f} images/sec")

        layer_start_time = time.time()  # Start timer for the layer
        out = self.relu1.forward(out)
        layer_time = time.time() - layer_start_time
        images_per_second = imgs / layer_time
        if self.first:
            print(f"Layer: {self.relu1.__class__.__name__}, Time: {layer_time:.4f}s, Performance: {images_per_second:.2f} images/sec")

        layer_start_time = time.time()  # Start timer for the layer
        out = self.conv2.forward(out)
        layer_time = time.time() - layer_start_time
        images_per_second = imgs / layer_time
        if self.first:
            print(f"Layer: {self.conv2.__class__.__name__}, Time: {layer_time:.4f}s, Performance: {images_per_second:.2f} images/sec")

        if self.use_projection:
            layer_start_time = time.time()  # Start timer for the layer
            identity = self.projection.forward(x)
            layer_time = time.time() - layer_start_time
            images_per_second = imgs / layer_time
            if self.first:
                print(f"Layer: {self.projection.__class__.__name__}, Time: {layer_time:.4f}s, Performance: {images_per_second:.2f} images/sec")
        else:
            identity = x

        self.output = [out[i] + identity[i] for i in range(len(out))]  # elementwise add

        layer_start_time = time.time()  # Start timer for the layer
        self.output = self.relu2.forward(self.output)
        layer_time = time.time() - layer_start_time
        if self.first:
            print(f"Layer: {self.relu2.__class__.__name__}, Time: {layer_time:.4f}s, Performance: {images_per_second:.2f} images/sec")
        self.first=False
        return self.output
    
    def backward(self, grad_output, learning_rate):
        grad = self.relu2.backward(grad_output, learning_rate)

        # Save original inputs and projection if needed
        if self.use_projection:
            identity = self.projection.forward(self.input)
        else:
            identity = self.input

        grad_main = self.conv2.backward(grad, learning_rate)
        grad_main = self.relu1.backward(grad_main, learning_rate)
        grad_main = self.conv1.backward(grad_main, learning_rate)

        if self.use_projection:
            grad_proj = self.projection.backward(grad, learning_rate)
        else:
            grad_proj = grad

        grad_input = [grad_main[i] + grad_proj[i] for i in range(len(grad_main))]  # elementwise add
        return grad_input


class ResNet18_CIFAR100:
    def __init__(self, use_im2col=False):
        print("Building ResNet18 for CIFAR-100")
        self.layers = []

        # Initial conv
        self.layers.append(Conv2D_np(3, 64, kernel_size=3, stride=1, padding=1, use_im2col=use_im2col))
        self.layers.append(ReLU_np())

        # Residual blocks
        self._make_layer(64, 64, 2, stride=1, use_im2col=use_im2col)
        self._make_layer(64, 128, 2, stride=2, use_im2col=use_im2col)
        self._make_layer(128, 256, 2, stride=2, use_im2col=use_im2col)
        self._make_layer(256, 512, 2, stride=2, use_im2col=use_im2col)

        # Global average pooling
        self.layers.append(GlobalAvgPool2D())

        # Flatten + Dense
        self.layers.append(Flatten_np())
        self.layers.append(Dense_np(512, 100))
        self.layers.append(Softmax_np())

    def _make_layer(self, in_channels, out_channels, blocks, stride, use_im2col):
        strides = [stride] + [1] * (blocks - 1)
        for s in strides:
            block = BasicBlock(in_channels, out_channels, s, use_im2col=use_im2col)
            self.layers.append(block)
            in_channels = out_channels  # For next block

    def forward(self, x, iter=1):
        for layer in self.layers:
            layer_start_time = time.time()  # Start timer for the layer
            x = layer.forward(x)
            layer_time = time.time() - layer_start_time
            if iter == 0 and layer.__class__.__name__ != 'BasicBlock':
                # Only print for the first iteration
                batch_size = x.shape[0]
                # Calculate performance
                images_per_second = x.shape[0] / layer_time
                print(f"Layer: {layer.__class__.__name__}, Time: {layer_time:.4f}s, Performance: {images_per_second:.2f} images/sec")
        return x

    def backward(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)
        return grad_output
