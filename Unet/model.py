import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable
        x = np.array(utils.generate_filter(band_start, band_end, size))
        self.base = nn.Parameter(torch.tensor(x).float(), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(x)), requires_grad=False)

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + utils.norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


class DctFilter(nn.Module):
    def __init__(self, size):
        super(DctFilter, self).__init__()
        dct_mat = utils.DCT_mat(size)
        self._DCT_all = nn.Parameter(torch.tensor(dct_mat).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(dct_mat).float(), 0, 1), requires_grad=False)
        low_filter = Filter(size, 0, size // 16)
        middle_filter = Filter(size, size // 16, size // 8)
        high_filter = Filter(size, size // 8, size)
        all_filter = Filter(size, 0, size * 2)
        # self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])
        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter])

    def forward(self, x):
        x_freq = self._DCT_all @ x @ self._DCT_all_T
        y_list = []
        for i in range(3):
            x_pass = self.filters[i](x_freq)
            y = self._DCT_all_T @ x_pass @ self._DCT_all
            y_list.append(y)
        out = torch.cat(y_list, dim=1)
        return out


class SobelConv(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(SobelConv, self).__init__()
        filter_x, filter_y = utils.get_sobel(in_chan=in_chan, out_chan=1)
        filter_x = torch.from_numpy(filter_x)
        filter_y = torch.from_numpy(filter_y)
        filter_x = nn.Parameter(filter_x, requires_grad=False)
        filter_y = nn.Parameter(filter_y, requires_grad=False)
        conv_x = nn.Conv2d(in_channels=in_chan, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        conv_y = nn.Conv2d(in_channels=in_chan, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        conv_x.weight = filter_x
        conv_y.weight = filter_y
        self.sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(1))
        self.sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(1))
        self.conv = nn.Conv2d(in_channels=3, out_channels=out_chan, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, input):
        g_x = self.sobel_x(input)
        g_y = self.sobel_y(input)
        g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
        x = torch.sigmoid(g) * input
        x = self.conv(x)
        return x


class BayarConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BayarConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1), requires_grad=True)

    def bayarConstraint(self):
        self.kernel.data = self.kernel.permute(2, 0, 1)
        self.kernel.data = torch.div(self.kernel.data, self.kernel.data.sum(0))
        self.kernel.data = self.kernel.permute(1, 2, 0)
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        x = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, mid_channels=None)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class VideoInpaintingDetectionNet(nn.Module):
    def __init__(self, bilinear=True, sobel=True, dct=True, bayarconv=True):
        super(VideoInpaintingDetectionNet, self).__init__()
        self.sobel = sobel
        self.dct = dct
        self.bayarconv = bayarconv
        self.bayar_conv = BayarConv(3, 9)
        self.sobel_conv = SobelConv(3, 9)
        self.dct_conv = DctFilter(512)

        self.inc = DoubleConv(27, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)

    def forward(self, x):
        f1 = self.sobel_conv(x)
        f2 = self.dct_conv(x)
        f3 = self.bayar_conv(x)
        f = torch.cat([f1, f2, f3], dim=1)

        x1 = self.inc(f)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        out = torch.sigmoid(x)

        return out


if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = VideoInpaintingDetectionNet(bilinear=True, sobel=True, dct=True, bayarconv=False)
    model = model.to(device)
    input = torch.rand(size=[1, 3, 512, 512]).to(device)
    print(input.size())
    output = model(input)
    print(output.size())
