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
https://github.com/tensorflow/models/blob/900b1e078e9a2866a29ac924946301ef0fe3b737/research/slim/nets/resnet_utils.py

Contains building blocks for various versions of Residual Networks.

Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

More variants were introduced in:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

We can obtain different ResNet variants by changing the network depth, width,
and form of residual unit. This module implements the infrastructure for
building them. Concrete ResNet units and full ResNet networks are implemented in
the accompanying resnet_v2.py module.
"""
import collections
import torch.nn as nn
import torch.nn.functional as F


class Block(collections.namedtuple('Block', ['unit_layer', 'args'])):
    """A named tuple describing a ResNet block.
    Its parts are:
        unit_layer: The ResNet unit layer which takes as input a `Tensor` and
            returns another `Tensor` with the output of the ResNet unit when
            forwarded.
        args: A list of length equal to the number of units in the `Block`. The
            list contains one (depth, depth_bottleneck, stride) tuple for each
            unit in the block to serve as argument to unit_layer.
    """
    pass


def subsample(inputs, factor):
    """Subsamples the input along the spatial dimensions.

    Args:
        inputs: A `Tensor` of size [batch, channels, height_in, width_in].
        factor: The subsampling factor.

    Returns:
        output: A `Tensor` of size [batch, channels, height_out, width_out] with
            the input, either intact (if factor == 1) or subsampled
            (if factor > 1).
    """
    if factor == 1:
        return inputs
    else:
        return F.max_pool2d(inputs, [1, 1], stride=factor)


class Subsample(nn.Module):
    def __init__(self, factor):
        super(Subsample, self).__init__()
        self.factor = factor

    def forward(self, inputs):
        return subsample(inputs, self.factor)


def conv2d_same(in_dim, num_outputs, kernel_size, stride, rate=1,
                normalizer_layer=None, activation_layer=None):
    """Strided 2-D convolution with 'SAME' padding.

    When stride > 1, then we do explicit zero-padding, followed by nn.Conv2d.

    Args:
        in_dim: An integer, the number of input channels.
        num_outputs: An integer, the number of output filters.
        kernel_size: An int with the kernel_size of the filters.
        stride: An integer, the output stride.
        rate: An integer, rate for atrous convolution.
        normalizer_layer: Any normalization layer.
        activation_layer: Any activation layer.
    Returns:
        An nn.Sequential object containing all layers specified.
    """
    layers = []

    if stride == 1:
        layers.append(nn.Conv2d(in_dim, num_outputs, kernel_size, stride=1, padding=1, dilation=rate))
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        layers.append(nn.ZeroPad2d((pad_beg, pad_end, pad_beg, pad_end)))
        layers.append(nn.Conv2d(in_dim, num_outputs, kernel_size, stride=stride, dilation=rate))

    if normalizer_layer is not None:
        layers.append(normalizer_layer)

    if activation_layer is not None:
        layers.append(activation_layer)

    return nn.Sequential(*layers)


def stack_blocks_dense(in_dim, blocks, output_stride=None, store_non_strided_activations=False, outputs_collections=None):
    """Stacks ResNet `Blocks` and controls output feature density.

    This function allows the user to explicitly control the ResNet
    output_stride, which is the ratio of the input to output spatial resolution.
    This is useful for dense prediction tasks such as semantic segmentation or
    object detection.

    Most ResNets consist of 4 ResNet blocks and subsample the activations by a
    factor of 2 when transitioning between consecutive ResNet blocks. This
    results to a nominal ResNet output_stride equal to 8. If we set the
    output_stride to half the nominal network stride (e.g., output_stride=4),
    then we compute responses twice.

    Control of the output feature density is implemented by atrous convolution.

    Args:
        net: A `Tensor` of size [batch, height, width, channels].
        blocks: A list of length equal to the number of ResNet `Blocks`. Each
            element is a ResNet `Block` object describing the units in the
            `Block`.
        output_stride: If `None`, then the output will be computed at the
            nominal network stride. If output_stride is not `None`, it specifies
            the requested ratio of input to output spatial resolution, which
            needs to be equal to the product of unit strides from the start up
            to some level of the ResNet. For example, if the ResNet employs
            units with strides 1, 2, 1, 3, 4, 1, then valid values for the
            output_stride are 1, 2, 6, 24 or None (which is equivalent to
            output_stride=24).
        store_non_strided_activations: If True, we compute non-strided
            (undecimated) activations at the last unit of each block and store
            them in the `outputs_collections` before subsampling them. This
            gives us access to higher resolution intermediate activations which
            are useful in some dense prediction problems but increases 4x the
            computation and memory cost at the last unit of each block.
        outputs_collections: Dictionary to add the ResNet block outputs.

    Returns:
        An OrderedDict of nn.Sequential instances of ResNet layers.

    Raises:
        ValueError: If the target output_stride is not valid.
    """
    # The current_stride variable keeps track of the effective stride of the
    # activations. This allows us to invoke atrous convolution whenever applying
    # the next residual unit would result in the activations having stride
    # larger than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    if outputs_collections is not None:
        def make_hook(name):
            def hook(model, input, output):
                outputs_collections[name] = output

            return hook

    layers = collections.defaultdict(list)
    for j, block in enumerate(blocks, start=1):
        block_name = 'block' + str(j)
        block_stride = 1
        for i, unit in enumerate(block.args):
            if store_non_strided_activations and i == len(block.args) - 1:
                # Move stride from the block's last unit to the end of the
                # block.
                block_stride = unit.get('stride', 1)
                unit = dict(unit, stride=1)

            # If we have reached the target output_stride, then we need to
            # employ atrous convolution with stride=1 and multiply the atrous
            # rate by the current unit's stride for use in subsequent layers.
            if output_stride is not None and current_stride == output_stride:
                layers[block_name].append(block.unit_layer(in_dim, rate=rate,
                                                           **dict(unit, stride=1
                                                                  )))
                rate *= unit.get('stride', 1)
            else:
                layers[block_name].append(block.unit_layer(in_dim, rate=1,
                                                           **unit))
                current_stride *= unit.get('stride', 1)
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be '
                                     'reached.')

            in_dim = unit['depth']

        if outputs_collections is not None:
            layers[block_name][-1].register_forward_hook(make_hook(block_name))

        if output_stride is not None and current_stride == output_stride:
            rate *= block_stride
        else:
            layers[block_name].append(Subsample(block_stride))
            current_stride *= block_stride
            if output_stride is not None and current_stride > output_stride:
                raise ValueError('The target output_stride cannot be reached.')

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')

    return collections.OrderedDict({
        k: nn.Sequential(*layers[k]) for k in sorted(layers.keys())
    })
