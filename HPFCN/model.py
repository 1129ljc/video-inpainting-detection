import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import resnet_v2

FILTERS = {
    'd1': [
        np.array([[0., 0., 0.],
                  [0., -1., 0.],
                  [0., 1., 0.]]),
        np.array([[0., 0., 0.],
                  [0., -1., 1.],
                  [0., 0., 0.]]),
        np.array([[0., 0., 0.],
                  [0., -1., 0.],
                  [0., 0., 1.]])],
    'd2': [
        np.array([[0., 1., 0.],
                  [0., -2., 0.],
                  [0., 1., 0.]]),
        np.array([[0., 0., 0.],
                  [1., -2., 1.],
                  [0., 0., 0.]]),
        np.array([[1., 0., 0.],
                  [0., -2., 0.],
                  [0., 0., 1.]])],
    'd3': [
        np.array([[0., 0., 0., 0., 0.],
                  [0., 0., -1., 0., 0.],
                  [0., 0., 3., 0., 0.],
                  [0., 0., -3., 0., 0.],
                  [0., 0., 1., 0., 0.]]),
        np.array([[0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [0., -1., 3., -3., 1.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.]]),
        np.array([[0., 0., 0., 0., 0.],
                  [0., -1., 0., 0., 0.],
                  [0., 0., 3., 0., 0.],
                  [0., 0., 0., -3., 0.],
                  [0., 0., 0., 0., 1.]])],
    'd4': [
        np.array([[0., 0., 1., 0., 0.],
                  [0., 0., -4., 0., 0.],
                  [0., 0., 6., 0., 0.],
                  [0., 0., -4., 0., 0.],
                  [0., 0., 1., 0., 0.]]),
        np.array([[0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.],
                  [1., -4., 6., -4., 1.],
                  [0., 0., 0., 0., 0.],
                  [0., 0., 0., 0., 0.]]),
        np.array([[1., 0., 0., 0., 0.],
                  [0., -4., 0., 0., 0.],
                  [0., 0., 6., 0., 0.],
                  [0., 0., 0., -4., 0.],
                  [0., 0., 0., 0., 1.]])],
}


def get_filter(filters):
    return np.repeat(filters[np.newaxis, np.newaxis, :, :], 3, axis=0)


def bilinear_upsample_weights(factor, out_channels, in_channels):
    """
    Create weights matrix for transposed convolution with bilinear filter initialization.
    """
    filter_size = 2 * factor - factor % 2
    center = (factor - 1) if filter_size % 2 == 1 else factor - 0.5
    og = np.ogrid[:filter_size, :filter_size]
    upsample_kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weights = np.tile(upsample_kernel[np.newaxis, np.newaxis, :, :], (in_channels, out_channels, 1, 1))
    return weights


class HighPassBlock(nn.Module):
    def __init__(self, filter_type='d1', filter_trainable=True):
        super(HighPassBlock, self).__init__()
        assert filter_type in ['random', 'd1', 'd2', 'd3', 'd4'], print('filter_type error')
        self.filter_type = filter_type
        self.filter_trainable = filter_trainable
        if self.filter_type in ['random', 'd1', 'd2']:
            self.kernel_size = 3
        elif self.filter_type in ['d3', 'd4']:
            self.kernel_size = 5
        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(self.kernel_size, self.kernel_size), stride=(1, 1), padding=(1, 1), groups=3)
        self.conv2d_2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(self.kernel_size, self.kernel_size), stride=(1, 1), padding=(1, 1), groups=3)
        self.conv2d_3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(self.kernel_size, self.kernel_size), stride=(1, 1), padding=(1, 1), groups=3)
        if self.filter_type == 'random':
            nn.init.xavier_uniform_(self.conv2d_1.weight)
            nn.init.xavier_uniform_(self.conv2d_2.weight)
            nn.init.xavier_uniform_(self.conv2d_2.weight)
        elif self.filter_type[0] == 'd':
            filters1 = get_filter(FILTERS[self.filter_type][0])
            filters2 = get_filter(FILTERS[self.filter_type][1])
            filters3 = get_filter(FILTERS[self.filter_type][2])
            filters1 = torch.from_numpy(filters1).float()
            filters2 = torch.from_numpy(filters2).float()
            filters3 = torch.from_numpy(filters3).float()
            filters1 = nn.Parameter(filters1, requires_grad=True)
            filters2 = nn.Parameter(filters2, requires_grad=True)
            filters3 = nn.Parameter(filters3, requires_grad=True)
            self.conv2d_1.weight = filters1
            self.conv2d_2.weight = filters2
            self.conv2d_3.weight = filters3

    def forward(self, x):
        x1 = self.conv2d_1(x)
        x2 = self.conv2d_2(x)
        x3 = self.conv2d_3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Encoder(nn.Module):
    def __init__(self, in_dim=9):
        super(Encoder, self).__init__()
        self.in_dim = in_dim
        blocks = [
            resnet_v2.resnet_v2_block(base_depth=32, num_units=2, stride=2),
            resnet_v2.resnet_v2_block(base_depth=64, num_units=2, stride=2),
            resnet_v2.resnet_v2_block(base_depth=128, num_units=2, stride=2),
            resnet_v2.resnet_v2_block(base_depth=256, num_units=2, stride=2),
        ]
        self.model = resnet_v2._resnet_v2(self.in_dim, blocks, num_classes=None, global_pool=False, output_stride=None, include_root_block=False, spatial_squeeze=True)

    def forward(self, x):
        x = self.model(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # self.convtranspose2d1 = nn.ConvTranspose2d(in_channels=1024, out_channels=64, kernel_size=(8, 8), stride=(4, 4), padding=(2, 2))
        # self.convtranspose2d2 = nn.ConvTranspose2d(in_channels=64, out_channels=4, kernel_size=(8, 8), stride=(4, 4), padding=(2, 2))
        # filters1 = bilinear_upsample_weights(4, 64, 1024)
        # filters1 = torch.from_numpy(filters1).float()
        # filters1 = nn.Parameter(filters1, requires_grad=True)
        # self.convtranspose2d1.weight = filters1
        # filters2 = bilinear_upsample_weights(4, 4, 64)
        # filters2 = torch.from_numpy(filters2).float()
        # filters2 = nn.Parameter(filters2, requires_grad=True)
        # self.convtranspose2d2.weight = filters2
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.double_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up1(x)
        x = self.double_conv1(x)
        x = self.up2(x)
        x = self.double_conv2(x)
        return x


class Decision(nn.Module):
    def __init__(self):
        super(Decision, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True)
        )
        self.dec = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.size())
        # x = self.conv(x)
        # print(x.size())
        # x = self.softmax(x)
        # print(x.size())
        # map_loss = x[:, 1, :, :]
        # map_loss = torch.unsqueeze(map_loss, dim=1)
        # map = torch.max(x, dim=1)[1]
        # map = torch.unsqueeze(map, dim=1)
        x = self.conv(x)
        x = self.dec(x)
        return x


class HighPassResNetV2(nn.Module):
    def __init__(self, filter_type):
        super(HighPassResNetV2, self).__init__()
        self.high_pass_residuals = HighPassBlock(filter_type=filter_type, filter_trainable=True)
        self.encoder = Encoder(in_dim=9)
        self.decoder = Decoder()
        self.decision = Decision()

    def forward(self, x):
        x = self.high_pass_residuals(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.decision(x)
        return x


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCELoss(reduction='none')

    def forward(self, logits, label):
        logits = logits.float()
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha
        pt = torch.where(label == 1, logits, 1 - logits)
        ce_loss = self.crit(logits, label)
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = HighPassResNetV2(filter_type='d4').to(device)
    print(model)
    input = torch.rand(size=[4, 3, 224, 224]).to(device)
    map = model(input)
    print(map.size())
