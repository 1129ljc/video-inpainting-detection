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
https://github.com/tensorflow/models/blob/900b1e078e9a2866a29ac924946301ef0fe3b737/research/slim/nets/resnet_v2.py

Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import resnet_utils

model_urls = {
    'resnet_v2_50': ''
}


class Bottleneck(nn.Module):
    """Bottleneck residual unit variant with BN before convolutions.

    This is the full preactivation residual unit variant proposed in [2]. See
    Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
    variant which has an extra bottleneck layer.

    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.

    Args:
        in_dim: The number of input channels.
        depth: The depth of the ResNet unit output.
        depth_bottleneck: The depth of the bottleneck layers.
        stride: The ResNet unit's stride. Determines the amount of downsampling
            of the units output compared to its input.
        rate: An integer, rate for atrous convolution.
    """

    def __init__(self, in_dim, depth, depth_bottleneck, stride, rate=1):
        super(Bottleneck, self).__init__()
        self.depth = depth
        self.stride = stride
        self.rate = rate

        self.preact = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU()
        )

        self.shortcut = nn.Conv2d(in_dim, depth, [1, 1])

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, depth_bottleneck, [1, 1], stride=1),
            nn.BatchNorm2d(depth_bottleneck),
            nn.ReLU()
        )
        self.conv2 = resnet_utils.conv2d_same(
            depth_bottleneck,
            depth_bottleneck,
            3,
            stride,
            rate=rate,
            normalizer_layer=nn.BatchNorm2d(depth_bottleneck),
            activation_layer=nn.ReLU()
        )
        self.conv3 = nn.Conv2d(depth_bottleneck, depth, [1, 1], stride=1)

    def forward(self, inputs):
        depth_in = inputs.size(1)
        preact = self.preact(inputs)
        if self.depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, self.stride)
        else:
            shortcut = self.shortcut(preact)
        residual = self.conv1(preact)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        output = shortcut + residual
        return output


class ResNetv2(nn.Module):
    """Base class for v2 (preactivation) ResNet models.

    See the resnet_v2_*() methods for specific model instantiations, obtained by
    selecting different block instantiations that produce ResNets of various
    depths.

    Training for image classification on Imagenet is usually done with
    [224, 224] inputs, resulting in [7, 7] feature maps at the output of the
    last ResNet block for the ResNets defined in [1] that have nominal stride
    equal to 32. However, for dense prediction tasks we advise that one uses
    inputs with spatial dimensions that are multiples of 32 plus 1, e.g.,
    [321, 321]. In this case the feature maps at the ResNet output will have
    spatial shape
    [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1] and
    corners exactly aligned with the input image corners, which greatly
    facilitates alignment of the features to the image. Using as input
    [225, 225] images results in [8, 8] feature maps at the output of the last
    ResNet block. For dense prediction tasks, the ResNet needs to run in
    fully-convolutional (FCN) mode and global_pool needs to be set to False.
    The ResNets in [1, 2] all have nominal stride equal to 32 and a good choice
    in FCN mode is to use output_stride=16 in order to increase the density of
    the computed features at small computational and memory overhead, cf.
    http://arxiv.org/abs/1606.00915.

    Args:
        in_dim: The number of input channels.
        blocks: A list of length equal to the number of ResNet blocks. Each
            element is a resnet_utils.Block object describing the units in the
            block.
        num_classes: Number of predicted classes for classification tasks.
            If 0 or None, we return the features before the logit layer.
        global_pool: If True, we perform global average pooling before computing
            the logits. Set to True for image classification, False for dense
            prediction.
        output_stride: If None, then the output will be computed at the nominal
            network stride. If output_stride is not None, it specifies the
            requested ratio of input to output spatial resolution.
        include_root_block: If True, include the initial convolution followed by
            max-pooling, if False excludes it. If excluded, `inputs` should be
            the results of an activation-less convolution.
        spatial_squeeze: if True, logits is of shape [B, C], if false logits is
            of shape [B, C, 1, 1], where B is batch_size and C is number of
            classes. To use this parameter, the input images must be smaller
            than 300x300 pixels, in which case the output logit layer does not
            contain spatial information and can be removed.

    Returns:
        out: A rank-4 tensor of size
            [batch, channels_out, height_out, width_out]. If global_pool is
            False, then height_out and width_out are reduced by a factor of
            output_stride compared to the respective height_in and width_in,
            else both height_out and width_out equal one. If num_classes is 0 or
            None, then net is the output of the last ResNet block, potentially
            after global average pooling. If num_classes is a non-zero integer,
            net contains the pre-softmax activations.
        end_points: A dictionary from components of the network to the
            corresponding activation.

    Raises:
        ValueError: If the target output_stride is not valid.
    """

    def __init__(self, in_dim, blocks, num_classes=None, global_pool=True, output_stride=None, include_root_block=True, spatial_squeeze=True):
        super(ResNetv2, self).__init__()

        self.include_root_block = include_root_block
        self.output_stride = output_stride
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.spatial_squeeze = spatial_squeeze

        if include_root_block:
            if output_stride is not None:
                if output_stride % 4 != 0:
                    raise ValueError('The output_stride needs to be a multiplen of 4.')
                self.output_stride /= 4
            # We do not include batch normalization or activation functions in
            # conv1 because the first ResNet unit will perform these. Cf.
            # Appendix of [2].
            self.conv1 = resnet_utils.conv2d_same(in_dim, 64, 7, stride=2)
            self.pool1 = nn.MaxPool2d([3, 3], stride=2)
        self.end_points = dict()
        self.layer_names = []
        layers = resnet_utils.stack_blocks_dense(64 if include_root_block else in_dim, blocks, output_stride, outputs_collections=self.end_points)
        for layer in layers:
            self.layer_names.append(layer)
            setattr(self, layer, layers[layer])
        # This is needed because the pre-activation variant does not have batch
        # normalization or activation functions in the residual unit output. See
        # Appendix of [2].
        self.postnorm = nn.Sequential(
            nn.BatchNorm2d(blocks[-1].args[-1]['depth']),
            nn.ReLU()
        )

        if num_classes:
            self.logits = nn.Conv2d(blocks[-1].args[-1]['depth'], num_classes, [1, 1])

    def forward(self, inputs):
        out = inputs
        if self.include_root_block:
            out = self.conv1(out)
            out = self.pool1(out)

        for layer_name in self.layer_names:
            out = getattr(self, layer_name)(out)

        out = self.postnorm(out)

        if self.global_pool:
            out = torch.mean(out, (2, 3), keepdim=True)
            self.end_points['global_pool'] = out
        if self.num_classes:
            out = self.logits(out)
            self.end_points['logits'] = out
            if self.spatial_squeeze:
                out = out.squeeze(2).squeeze(2)
                self.end_points['spatial_squeeze'] = out
            self.end_points['predictions'] = F.softmax(out)

        return out


def resnet_v2_block(base_depth, num_units, stride):
    """Helper function for creating a resnet_v2 bottleneck block.
    Args:
        base_depth: The depth of the bottleneck layer for each unit.
        num_units: The number of units in the block.
        stride: The stride of the block, implemented as a stride in the last
            unit. All other units have stride = 1.
    Returns:
        A resnet_v2 bottleneck block.
    """
    return resnet_utils.Block(Bottleneck, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1
    }] * (num_units - 1) + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride
    }])


def _resnet_v2(in_dim, blocks, num_classes, global_pool, output_stride, spatial_squeeze, include_root_block):
    return ResNetv2(in_dim, blocks, num_classes, global_pool, output_stride, include_root_block=include_root_block, spatial_squeeze=spatial_squeeze)


def resnet_v2_50(in_dim, pretrained=False, progress=True, num_classes=None, global_pool=True, output_stride=None, spatial_squeeze=True):
    """ResNet-50 model of [1]. See ResNetv2 for arg and return description."""
    blocks = [
        resnet_v2_block(base_depth=64, num_units=3, stride=2),
        resnet_v2_block(base_depth=128, num_units=4, stride=2),
        resnet_v2_block(base_depth=256, num_units=6, stride=2),
        resnet_v2_block(base_depth=512, num_units=3, stride=1),
    ]
    model = _resnet_v2(in_dim, blocks, num_classes, global_pool, output_stride,
                       spatial_squeeze)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet_v2_50'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet_v2_101(in_dim, pretrained=False, progress=True, num_classes=None, global_pool=True, output_stride=None, spatial_squeeze=True):
    """ResNet-101 model of [1]. See ResNetv2 for arg and return description."""
    blocks = [
        resnet_v2_block(base_depth=64, num_units=3, stride=2),
        resnet_v2_block(base_depth=128, num_units=4, stride=2),
        resnet_v2_block(base_depth=256, num_units=23, stride=2),
        resnet_v2_block(base_depth=512, num_units=3, stride=1),
    ]
    model = _resnet_v2(in_dim, blocks, num_classes, global_pool, output_stride,
                       spatial_squeeze)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet_v2_101'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet_v2_152(in_dim, pretrained=False, progress=True, num_classes=None, global_pool=True, output_stride=None, spatial_squeeze=True):
    """ResNet-152 model of [1]. See ResNetv2 for arg and return description."""
    blocks = [
        resnet_v2_block(base_depth=64, num_units=3, stride=2),
        resnet_v2_block(base_depth=128, num_units=8, stride=2),
        resnet_v2_block(base_depth=256, num_units=36, stride=2),
        resnet_v2_block(base_depth=512, num_units=3, stride=1),
    ]
    model = _resnet_v2(in_dim, blocks, num_classes, global_pool, output_stride,
                       spatial_squeeze)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet_v2_152'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet_v2_200(in_dim, pretrained=False, progress=True, num_classes=None, global_pool=True, output_stride=None, spatial_squeeze=True):
    """ResNet-200 model of [1]. See ResNetv2 for arg and return description."""
    blocks = [
        resnet_v2_block(base_depth=64, num_units=3, stride=2),
        resnet_v2_block(base_depth=128, num_units=24, stride=2),
        resnet_v2_block(base_depth=256, num_units=36, stride=2),
        resnet_v2_block(base_depth=512, num_units=3, stride=1),
    ]
    model = _resnet_v2(in_dim, blocks, num_classes, global_pool, output_stride, spatial_squeeze)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet_v2_200'], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet_v2_encoder(in_dim):
    blocks = [
        resnet_v2_block(base_depth=32, num_units=2, stride=2),
        resnet_v2_block(base_depth=64, num_units=2, stride=2),
        resnet_v2_block(base_depth=128, num_units=2, stride=2),
        resnet_v2_block(base_depth=256, num_units=2, stride=2),
    ]
    model = _resnet_v2(in_dim, blocks, num_classes=None, global_pool=False, output_stride=None, include_root_block=False, spatial_squeeze=True)
    return model


if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = resnet_v2_encoder(in_dim=9).to(device)
    print(model)
    input = torch.rand(size=[1, 9, 224, 224]).to(device)
    output = model(input)
    print(output.size())
