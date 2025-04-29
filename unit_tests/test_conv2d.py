import numpy as np
from modules.conv2d import Conv2D

import numpy as np

def test_conv2d_forward_multiple_outputs():
    # Create Conv2D layer: 1 input channel, 3 output channels, 3x3 kernel, no padding, stride 1
    conv = Conv2D(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0, conv_algo=2)
    batch = 2
    # Input: 1 image, 1 channel, 5x5 values from 0 to 24
    input_image = np.arange(25*batch, dtype=np.float32).reshape(batch, 1, 5, 5)

    # All kernels initialized to the same all-ones pattern for test simplicity
    conv.kernels = np.ones((3, 1, 3, 3), dtype=np.float32)
    conv.biases = np.zeros(3, dtype=np.float32)

    # Compute expected output using naive convolution for all 3 channels
    expected_output = np.zeros((batch, 3, 3, 3), dtype=np.float32)
    for b in range(batch):  # batch
        for c in range(3):  # output channels
            for i in range(3):  # output height
                for j in range(3):  # output width
                    patch = input_image[b, 0, i:i+3, j:j+3]
                    expected_output[b, c, i, j] = np.sum(patch)  # all-ones kernel

    # Run forward pass
    output = conv.forward(input_image)

    # Assert
    assert np.allclose(output, expected_output), "Conv2D multi-output forward mismatch!"
    print("✅ Conv2D forward with 3 output channels passed!")

test_conv2d_forward_multiple_outputs()

