import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, use_instance_norm):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes) if use_instance_norm else nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes // 2))
        self.bn2 = (nn.InstanceNorm2d(int(out_planes / 2)) if use_instance_norm
                    else nn.BatchNorm2d(int(out_planes / 2)))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = (nn.InstanceNorm2d(int(out_planes / 4)) if use_instance_norm
                    else nn.BatchNorm2d(int(out_planes / 4)))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(nn.InstanceNorm2d(in_planes) if use_instance_norm
                                            else nn.BatchNorm2d(in_planes),
                                            nn.ReLU(True),
                                            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False))
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class HourGlass(nn.Module):
    def __init__(self, config):
        super(HourGlass, self).__init__()
        self.config = config

        self._generate_network(self.config.hg_depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.config.hg_num_features,
                                                      self.config.hg_num_features,
                                                      self.config.use_instance_norm))

        self.add_module('b2_' + str(level), ConvBlock(self.config.hg_num_features,
                                                      self.config.hg_num_features,
                                                      self.config.use_instance_norm))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.config.hg_num_features,
                                                               self.config.hg_num_features,
                                                               self.config.use_instance_norm))

        self.add_module('b3_' + str(level), ConvBlock(self.config.hg_num_features,
                                                      self.config.hg_num_features,
                                                      self.config.use_instance_norm))

    def _forward(self, level, inp):
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        if self.config.use_avg_pool:
            low1 = F.avg_pool2d(inp, 2)
        else:
            low1 = F.max_pool2d(inp, 2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x):
        return self._forward(self.config.hg_depth, x)


class FAN(nn.Module):
    def __init__(self, config=None):
        super(FAN, self).__init__()
        if config is None:
            self.config = FAN.create_config()
        else:
            self.config = config

        # Stem
        self.conv1 = nn.Conv2d(self.config.in_channel, 64, kernel_size=self.config.stem_conv_kernel_size,
                               stride=self.config.stem_conv_stride,
                               padding=self.config.stem_conv_kernel_size // 2)
        self.bn1 = nn.InstanceNorm2d(64) if self.config.use_instance_norm else nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128, self.config.use_instance_norm)
        self.conv3 = ConvBlock(128, 128, self.config.use_instance_norm)
        self.conv4 = ConvBlock(128, self.config.hg_num_features, self.config.use_instance_norm)

        # Hourglasses
        for hg_module in range(self.config.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(self.config))
            self.add_module('top_m_' + str(hg_module), ConvBlock(self.config.hg_num_features,
                                                                 self.config.hg_num_features,
                                                                 self.config.use_instance_norm))
            self.add_module('conv_last' + str(hg_module), nn.Conv2d(self.config.hg_num_features,
                                                                    self.config.hg_num_features,
                                                                    kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module),
                            nn.InstanceNorm2d(self.config.hg_num_features) if self.config.use_instance_norm
                            else nn.BatchNorm2d(self.config.hg_num_features))
            self.add_module('l' + str(hg_module), nn.Conv2d(self.config.hg_num_features,
                                                            self.config.num_landmarks,
                                                            kernel_size=1, stride=1, padding=0))

            if hg_module < self.config.num_modules - 1:
                self.add_module('bl' + str(hg_module), nn.Conv2d(self.config.hg_num_features,
                                                                 self.config.hg_num_features,
                                                                 kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(self.config.num_landmarks,
                                                                 self.config.hg_num_features,
                                                                 kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = self.conv2(F.relu(self.bn1(self.conv1(x)), True))
        if self.config.stem_pool_kernel_size > 1:
            if self.config.use_avg_pool:
                x = F.avg_pool2d(x, self.config.stem_pool_kernel_size)
            else:
                x = F.max_pool2d(x, self.config.stem_pool_kernel_size)
        x = self.conv3(x)
        x = self.conv4(x)

        outputs = []
        previous = x
        for i in range(self.config.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.config.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs

    @staticmethod
    def create_config(input_size=256, num_modules=2, hg_num_features=256, hg_depth=4,
                      use_avg_pool=False, use_instance_norm=False, stem_conv_kernel_size=7,
                      stem_conv_stride=2, stem_pool_kernel_size=2, num_landmarks=68, in_channel=3):
        return SimpleNamespace(input_size=input_size, num_modules=num_modules, hg_num_features=hg_num_features,
                               hg_depth=hg_depth, use_avg_pool=use_avg_pool, use_instance_norm=use_instance_norm,
                               stem_conv_kernel_size=stem_conv_kernel_size, stem_conv_stride=stem_conv_stride,
                               stem_pool_kernel_size=stem_pool_kernel_size, num_landmarks=num_landmarks, in_channel=in_channel)
