import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels, 
              kernel_size, stride, padding, dilation, groups, bias)
        self.register_buffer('_mask', torch.ones(self.weight.size()))

    def forward(self, x):
        ### Masked output
        weight = self.weight * self.mask
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    @property
    def mask(self):
        return Variable(self._mask)

    @mask.setter
    def mask(self, value):
        self._mask = value


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias=True)
        self.register_buffer('_mask', torch.ones(self.weight.size()))

    def forward(self, x):
        ### Masked output
        weight = self.weight * self.mask
        return F.linear(x, weight, self.bias)

    @property
    def mask(self):
        return Variable(self._mask)

    @mask.setter
    def mask(self, value):
        self._mask = value