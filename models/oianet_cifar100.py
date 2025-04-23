from modules.conv2d import Conv2D
from modules.relu import ReLU
from modules.maxpool2d import MaxPool2D
from modules.flatten import Flatten
from modules.dense import Dense
from modules.softmax import Softmax
from models.basemodel import BaseModel


class OIANET_CIFAR100(BaseModel):
    def __init__(self, use_im2col=False):
        print("Building OIANet for CIFAR-100")
        layers = [
            Conv2D(3, 32, kernel_size=3, stride=1, padding=1, use_im2col=use_im2col),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),

            Conv2D(32, 64, kernel_size=3, stride=1, padding=1, use_im2col=use_im2col),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),

            Conv2D(64, 128, kernel_size=3, stride=1, padding=1, use_im2col=use_im2col),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),

            Flatten(),
            Dense(128 * 4 * 4, 256),
            ReLU(),
            Dense(256, 100),
            Softmax()
        ]
        super().__init__(layers)