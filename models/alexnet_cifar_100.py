from modules.conv2d import Conv2D, Conv2D_np
from modules.relu import ReLU, ReLU_np
from modules.maxpool2d import MaxPool2D, MaxPool2D_np
from modules.flatten import Flatten, Flatten_np
from modules.dense import Dense, Dense_np
from modules.softmax import Softmax, Softmax_np
from modules.layer import Layer
def build_alexnet_for_cifar100():
    """
         return [
         Conv2D_np(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1),
         ReLU_np(),
         MaxPool2D_np(kernel_size=2, stride=2),  # 32x32 → 16x16

         Conv2D_np(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
         ReLU_np(),
         MaxPool2D_np(kernel_size=2, stride=2),  # 16x16 → 8x8
         Conv2D_np(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
         ReLU_np(),

         Conv2D_np(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
         ReLU_np(),
         Conv2D_np(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
         ReLU_np(),
         MaxPool2D_np(kernel_size=2, stride=2),  # 8x8 → 4x4

         Flatten_np(),
         Dense_np(8 * 4 * 4, 1024),
         ReLU_np(),
         Dense_np(1024, 512),
         ReLU_np(),
         Dense_np(512, 100),  # 100 classes for CIFAR-100
         Softmax_np()
     ]
    """
    return [
         Conv2D_np(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
         ReLU_np(),
         MaxPool2D_np(kernel_size=2, stride=2),  # 32x32 → 16x16
         Conv2D_np(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
         ReLU_np(),
         MaxPool2D_np(kernel_size=2, stride=2),  # 16x16 → 8x8
         Conv2D_np(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),
         ReLU_np(),

         Conv2D_np(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
         ReLU_np(),

         Conv2D_np(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
         ReLU_np(),
         MaxPool2D_np(kernel_size=2, stride=2),  # 8x8 → 4x4

         Flatten_np(),
         Dense_np(256 * 4 * 4, 1024),
         ReLU_np(),
         Dense_np(1024, 512),
         ReLU_np(),
         Dense_np(512, 100),  # 100 classes for CIFAR-100
         Softmax_np()
    ]
