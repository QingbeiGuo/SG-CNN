from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.model.utils.config import cfg
from lib.model.faster_rcnn.faster_rcnn import _fasterRCNN

from lib.model.faster_rcnn.layers import MaskedConv2d
from lib.model.faster_rcnn.layers import MaskedLinear

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb
from collections import OrderedDict
import re

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


'''model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-241335ed.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-6f0f7f60.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-4c113574.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-17b70270.pth',
}'''

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        #self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('conv1', MaskedConv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        #self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.add_module('conv2', MaskedConv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        #self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('conv', MaskedConv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class Self_GroupingDenseNetModel(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, in_shape=(3,244,244), num_classes=1000, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0):

        super(Self_GroupingDenseNetModel, self).__init__()
        in_channels, height, width = in_shape

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        #self.classifier = nn.Linear(num_features, num_classes)
        self.classifier = MaskedLinear(num_features, num_classes)

    def forward(self, x):

        features = self.features(x)
        out = F.relu(features, inplace=True)
        #out = F.dropout(out, p=0.20, training=self.training)
        out = F.adaptive_avg_pool2d(out,output_size=1)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

    def set_conv_mask(self, layer_index, centroids, data_to_centroids):
        convlayers = 0
        for module in self.modules():
            if module.__str__().startswith('MaskedConv2d'):
                if convlayers == layer_index:
                    module._group = data_to_centroids

                    for i in range(module.out_channels):
                        for j in range(module.in_channels):
                            #print(module._mask[i,j,:,:])
                            #print(centroids[data_to_centroids[i]][j])
                            module._mask[i,j,:,:] = centroids[data_to_centroids[i]][j] * module._mask[i,j,:,:]
                            #print(module._mask[i,j,:,:])
                convlayers = convlayers + 1

    def set_linear_mask(self, layer_index, centroids, data_to_centroids):
        linearlayers = 0
        for module in self.modules():
            if module.__str__().startswith('MaskedLinear'):
                if linearlayers == layer_index:
                    module._group = data_to_centroids

                    for i in range(module.out_features):
                        for j in range(module.in_features):
                            #print(module._mask[i,j])
                            #print(centroids[data_to_centroids[i]][j])
                            module._mask[i,j] = centroids[data_to_centroids[i]][j] * module._mask[i,j]
                            #print(module._mask[i,j])
                linearlayers = linearlayers + 1


def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Self_GroupingDenseNetModel(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet121']))
    return model


def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Self_GroupingDenseNetModel(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet169']))
    return model


def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Self_GroupingDenseNetModel(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet201']))
    return model


def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Self_GroupingDenseNetModel(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet161']))
    return model


class densenet(_fasterRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
    #self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
    #self.model_path = '/data/qingbeiguo/work/faster-rcnn.pytorch-pytorch-1.0/faster-rcnn.pytorch-pytorch-1.0/lib/model/faster_rcnn/pretrained_model/resnet101_caffe.pth'
    self.model_path = '/data/qingbeiguo/work/gcnn-Faster-rcnn/faster-rcnn.pytorch-pytorch-1.0-dense201-coco-600-G70/lib/model/faster_rcnn/pretrained_model/model_training_G70.pth'
    #self.dout_base_model = 896
    self.dout_base_model = 1792
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    densenet = densenet201()
    #print("densenet", densenet)

    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      densenet.load_state_dict({k:v for k,v in state_dict.items() if k in densenet.state_dict()})
#      densenet = densenet201(pretrained=True)
      print("densenet", densenet)

    # Build densenet.
    #feature = [conv0, norm0, relu0, pool0, block1, || trans1, block2, trans2, block3, || tran3, block4, norm]
    self.RCNN_base = nn.Sequential(*list(densenet.features._modules.values())[:-3])
    print("self.RCNN_base", self.RCNN_base)

    self.RCNN_top = nn.Sequential(*list(densenet.features._modules.values())[-3:])
    print("self.RCNN_top", self.RCNN_top)

    self.RCNN_cls_score = nn.Linear(1920, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(1920, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(1920, 4 * self.n_classes)

    # Fix blocks
    for p in self.RCNN_base[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base[1].parameters(): p.requires_grad=False

    print("cfg.RESNET.FIXED_BLOCKS", cfg.RESNET.FIXED_BLOCKS)
    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_base[8].parameters(): p.requires_grad=False
      #for p in self.RCNN_base[9].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_base[6].parameters(): p.requires_grad=False
      for p in self.RCNN_base[7].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_base[4].parameters(): p.requires_grad=False
      #for p in self.RCNN_base[5].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base.eval()
      self.RCNN_base[5].train()
      self.RCNN_base[6].train()
      self.RCNN_base[7].train()
      self.RCNN_base[8].train()
      #self.RCNN_base[9].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    out = self.RCNN_top(pool5)
    out = F.relu(out, inplace=True)
    out = F.adaptive_avg_pool2d(out, output_size=1)
    fc7 = out.view(out.size(0), -1)

    #fc7 = self.RCNN_top(F.relu(pool5, inplace=True)).mean(3).mean(2)
    #fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7
