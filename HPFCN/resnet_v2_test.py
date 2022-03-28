# Copyright 2020 Hoeseong Kim. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PyTorch adaptation of the TensorFlow version of ResNet v2. Original code:

"""
import numpy as np
import torch
import resnet_utils


def create_test_input(batch_size, channels, height, width):
    """Create test input tensor.
    Args:
        batch_size: The number of images per batch.
        channels: The number of channels per image.
        height: The height of each image.
        width: The width of each image.
    Returns:
        A constant `Tensor` with the mesh grid values along the spatial
        dimensions.
    """
    return torch.FloatTensor(
        np.tile(
            np.reshape(
                np.reshape(np.arange(height), [height, 1]) +
                np.reshape(np.arange(width), [1, width]),
                [1, 1, height, width]), [batch_size, channels, 1, 1])
        )


def testSubsampleThreeByThree():
    x = torch.arange(9, dtype=torch.float32).reshape((1, 1, 3, 3))
    x = resnet_utils.subsample(x, 2)
    expected = torch.FloatTensor([0, 2, 6, 8]).reshape((1, 1, 2, 2))
    assert torch.allclose(x, expected)


def testSubsampleFourByFour():
    x = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
    x = resnet_utils.subsample(x, 2)
    expected = torch.FloatTensor([0, 2, 8, 10]).reshape((1, 1, 2, 2))
    assert torch.allclose(x, expected)


def _testConv2DSame(n, n2, expected_output):
    # Input image.
    x = create_test_input(1, 1, n, n)

    # Convolution kernel.
    w = create_test_input(1, 1, 3, 3)

    # Note that y1, y2, and y4 of the original test code are not tested as
    # PyTorch does not support VALID padding for nn.Conv2D
    y3_conv = resnet_utils.conv2d_same(1, 1, 3, stride=2)
    y3_conv[1].weight.data = w
    y3_conv[1].bias.data = torch.zeros_like(y3_conv[1].bias.data)
    y3 = y3_conv(x)
    y3_expected = torch.FloatTensor(expected_output)
    y3_expected = y3_expected.reshape((1, 1, n2, n2))

    assert torch.allclose(y3, y3_expected)


def testConv2DSameEven():
    n, n2 = 4, 2
    _testConv2DSame(n, n2, [[14, 43], [43, 84]])


def testConv2DSameOdd():
    n, n2 = 5, 3
    _testConv2DSame(n, n2, [[14, 43, 34], [43, 84, 55], [34, 55, 30]])


if __name__ == '__main__':
    create_test_input()
