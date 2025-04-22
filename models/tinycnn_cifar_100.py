from modules.conv2d import Conv2D, Conv2D_np
from modules.relu import ReLU, ReLU_np
from modules.maxpool2d import MaxPool2D, MaxPool2D_np
from modules.flatten import Flatten, Flatten_np
from modules.dense import Dense, Dense_np
from modules.softmax import Softmax, Softmax_np
from modules.avgpool2d import GlobalAvgPool2D
from modules.layer import Layer
from models.basemodel import BaseModel


class TinyCNN(BaseModel):
    def __init__(self, use_im2col=False):
        print("Building TinyCNN for CIFAR-100")
        layers = [
            Conv2D_np(3, 32, kernel_size=3, stride=1, padding=1, use_im2col=use_im2col),
            ReLU_np(),
            Conv2D_np(32, 64, kernel_size=3, stride=1, padding=1, use_im2col=use_im2col),
            ReLU_np(),
            GlobalAvgPool2D(),
            Flatten_np(),
            Dense_np(64, 100),  # Output layer for 100 classes
            Softmax_np()
        ]
        super().__init__(layers)

    
    
