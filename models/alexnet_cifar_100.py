from modules.conv2d import Conv2D, Conv2D_np
from modules.relu import ReLU, ReLU_np
from modules.maxpool2d import MaxPool2D, MaxPool2D_np
from modules.flatten import Flatten, Flatten_np
from modules.dense import Dense, Dense_np
from modules.softmax import Softmax, Softmax_np
from modules.layer import Layer
from models.basemodel import BaseModel

class AlexNet_CIFAR100(BaseModel): 
     def __init__(self, use_im2col=False):
         print("Building AlexNet for CIFAR-100")
         layers = [] 
         layers.append(Conv2D_np(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, use_im2col=use_im2col))
         layers.append(ReLU_np())
         layers.append(MaxPool2D_np(kernel_size=2, stride=2))  # 32x32 → 16x16
         
         layers.append(Conv2D_np(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1, use_im2col=use_im2col))
         layers.append(ReLU_np())
         layers.append(MaxPool2D_np(kernel_size=2, stride=2))  # 16x16 → 8x8
         
         layers.append(Conv2D_np(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1, use_im2col=use_im2col))
         layers.append(ReLU_np())

         layers.append(Conv2D_np(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, use_im2col=use_im2col))
         layers.append(ReLU_np())

         layers.append(Conv2D_np(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, use_im2col=use_im2col))
         layers.append(ReLU_np())
         layers.append(MaxPool2D_np(kernel_size=2, stride=2))  # 8x8 → 4x4

         layers.append(Flatten_np())
         layers.append(Dense_np(256 * 4 * 4, 1024))
         layers.append(ReLU_np())
         
         layers.append(Dense_np(1024, 512))
         layers.append(ReLU_np())
         layers.append(Dense_np(512, 100))  # 100 classes for CIFAR-100
         layers.append(Softmax_np())

         super().__init__(layers)

