from modules.conv2d import Conv2D
from modules.relu import ReLU
from modules.maxpool2d import MaxPool2D
from modules.flatten import Flatten
from modules.dense import Dense
from modules.softmax import Softmax
from modules.avgpool2d import GlobalAvgPool2D
from models.basemodel import BaseModel


class TinyCNN(BaseModel):
    def __init__(self, use_im2col=False):
        print("Building TinyCNN for CIFAR-100")
        layers = [
            Conv2D(3, 32, kernel_size=3, stride=1, padding=1, use_im2col=use_im2col),
            ReLU(),
            Conv2D(32, 64, kernel_size=3, stride=1, padding=1, use_im2col=use_im2col),
            ReLU(),
            GlobalAvgPool2D(),
            Flatten(),
            Dense(64, 100),  # Output layer for 100 classes
            Softmax()
        ]
        super().__init__(layers)

    
    
