import math
import torch.nn as nn


# from .slimmable_ops import USBatchNorm2d, USConv2d, make_divisible
from utils.config import FLAGS


def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class USConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, depthwise=False, bias=True,
                 us=[True, True], ratio=[1, 1], block_idx=0, firstCNN =False):
        super(USConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.depthwise = depthwise
        self.in_channels_max = in_channels
        self.out_channels_max = out_channels
        self.width_mult = None
        self.us = us
        self.ratio = ratio

        self.firstCNN = firstCNN

        self.block_idx = block_idx
        self.blocksize_list = None


    def forward(self, input):

        self.width_mult_in = self.blocksize_list[self.block_idx] if not self.firstCNN else self.blocksize_list[self.block_idx-1]
        self.width_mult_out = self.blocksize_list[self.block_idx]

        if self.us[0]:
            self.in_channels = make_divisible(
                self.in_channels_max
                * self.width_mult_in
                / self.ratio[0]) * self.ratio[0]
        if self.us[1]:
            self.out_channels = make_divisible(
                self.out_channels_max
                * self.width_mult_out
                / self.ratio[1]) * self.ratio[1]
        self.groups = self.in_channels if self.depthwise else 1
        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(
            input, weight, bias, self.stride, self.padding,
            self.dilation, self.groups)
        # if getattr(FLAGS, 'conv_averaged', False):
        #     y = y * (max(self.in_channels_list) / self.in_channels)
        return y


class USBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, ratio=1, block_idx=0):
        super(USBatchNorm2d, self).__init__(
            num_features, affine=True, track_running_stats=False)
        self.num_features_max = num_features
        # for tracking performance during training
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(i, affine=False) for i in [
                make_divisible(
                    self.num_features_max * width_mult / ratio) * ratio
                for width_mult in FLAGS.width_mult_list]])
        self.ratio = ratio
        self.width_mult = None
        self.ignore_model_profiling = True

        self.block_idx = block_idx
        self.blocksize_list = None

    def forward(self, input):

        self.width_mult = self.blocksize_list[self.block_idx]

        weight = self.weight
        bias = self.bias
        c = make_divisible(
            self.num_features_max * self.width_mult / self.ratio) * self.ratio
        if self.width_mult in FLAGS.width_mult_list:
            idx = FLAGS.width_mult_list.index(self.width_mult)
            y = nn.functional.batch_norm(
                input,
                self.bn[idx].running_mean[:c],
                self.bn[idx].running_var[:c],
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        else:
            y = nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                weight[:c],
                bias[:c],
                self.training,
                self.momentum,
                self.eps)
        return y


class InvertedResidual(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio, block_idx, firstCNN_tag = False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.residual_connection = stride == 1 and inp == outp


        self.block_idx = block_idx
        self.firstCNN_tag = firstCNN_tag


        layers = []
        # expand
        expand_inp = inp * expand_ratio
        if expand_ratio != 1:
            layers += [
                USConv2d(
                    inp, expand_inp, 1, 1, 0, bias=False,
                    ratio=[1, expand_ratio], block_idx=self.block_idx, firstCNN = self.firstCNN_tag),
                USBatchNorm2d(expand_inp, ratio=expand_ratio, block_idx=self.block_idx),
                nn.ReLU6(inplace=True),
            ]
        # depthwise + project back
        if expand_ratio != 1:
            layers += [
                USConv2d(
                    expand_inp, expand_inp, 3, stride, 1, groups=expand_inp,
                    depthwise=True, bias=False,
                    ratio=[expand_ratio, expand_ratio], block_idx=self.block_idx),
                USBatchNorm2d(expand_inp, ratio=expand_ratio, block_idx=self.block_idx),

                nn.ReLU6(inplace=True),

                USConv2d(
                    expand_inp, outp, 1, 1, 0, bias=False,
                    ratio=[expand_ratio, 1], block_idx=self.block_idx),
                USBatchNorm2d(outp, block_idx=self.block_idx),
            ]
        else :
            layers += [
                USConv2d(
                    expand_inp, expand_inp, 3, stride, 1, groups=expand_inp,
                    depthwise=True, bias=False, 
                    ratio=[expand_ratio, expand_ratio], block_idx=self.block_idx, firstCNN = self.firstCNN_tag),
                USBatchNorm2d(expand_inp, ratio=expand_ratio, block_idx=self.block_idx),

                nn.ReLU6(inplace=True),

                USConv2d(
                    expand_inp, outp, 1, 1, 0, bias=False,
                    ratio=[expand_ratio, 1], block_idx=self.block_idx),
                USBatchNorm2d(outp, block_idx=self.block_idx),
            ]

        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_connection and (not self.firstCNN_tag):
            res = self.body(x)
            res += x
        else:
            res = self.body(x)
        return res


class Model(nn.Module):
    def __init__(self, num_classes=1000, input_size=224):
        super(Model, self).__init__()

        # setting of inverted residual blocks
        self.block_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        if FLAGS.dataset == 'cifar10':
            self.block_setting[2] = [6, 24, 2, 1]

        self.features = []

        width_mult = FLAGS.width_mult_range[-1]
        # width_mult = max(width_mult_list)
        idx = 0
        # head
        assert input_size % 32 == 0
        channels = make_divisible(32 * width_mult)
        self.outp = make_divisible(
            1280 * width_mult) if width_mult > 1.0 else 1280
        first_stride = 2
        self.features.append(
            nn.Sequential( 
                USConv2d(
                    3, channels, 3, first_stride, 1, bias=False,
                    us=[False, True], block_idx = idx),
                USBatchNorm2d(channels, block_idx = idx),
                nn.ReLU6(inplace=True))
        )
        idx = idx + 1
        
        # body
        for t, c, n, s in self.block_setting:
            outp = make_divisible(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        InvertedResidual(channels, outp, s, t, idx, firstCNN_tag = True))
                else:
                    self.features.append(
                        InvertedResidual(channels, outp, 1, t, idx))
                channels = outp
            idx = idx + 1

        # tail
        self.features.append(
            nn.Sequential(
                USConv2d(
                    channels, self.outp, 1, 1, 0, bias=False,
                    us=[True, False], block_idx = idx, firstCNN = True),
                nn.BatchNorm2d(self.outp),
                nn.ReLU6(inplace=True),
            )
        )
        avg_pool_size = input_size // 32
        self.features.append(nn.AvgPool2d(avg_pool_size))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(self.outp, num_classes))
        # if FLAGS.reset_parameters:
        #     self.reset_parameters()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.outp)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
