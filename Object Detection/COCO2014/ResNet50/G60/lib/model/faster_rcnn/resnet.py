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

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    #return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    return MaskedConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        '''m = OrderedDict()
        m['conv1'] = conv3x3(inplanes, planes, stride)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(planes, planes)
        m['bn2'] = nn.BatchNorm2d(planes)
        self.group1 = nn.Sequential(m)'''

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        #self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        #out = self.group1(x) + residual
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual

        #out = self.relu(out)
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        '''m  = OrderedDict()
        m['conv1'] = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        m['bn2'] = nn.BatchNorm2d(planes)
        m['relu2'] = nn.ReLU(inplace=True)
        m['conv3'] = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        m['bn3'] = nn.BatchNorm2d(planes * 4)
        self.group1 = nn.Sequential(m)'''

        '''self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)'''

        self.conv1 = MaskedConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = MaskedConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = MaskedConv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        #self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        #out = self.group1(x) + residual
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        out = out + residual

        #out = self.relu(out)
        out = F.relu(out)

        return out


class Self_GroupingResNetModel(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(Self_GroupingResNetModel, self).__init__()

        '''m = OrderedDict()
        m['conv1'] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m['bn1'] = nn.BatchNorm2d(64)
        m['relu1'] = nn.ReLU(inplace=True)
        m['maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1= nn.Sequential(m)'''
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        #self.avgpool = nn.Sequential(nn.AvgPool2d(7))
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        '''self.group2 = nn.Sequential(
            OrderedDict([
                ('fc', nn.Linear(512 * block.expansion, num_classes))
            ])
        )'''
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = MaskedLinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #x = self.group1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.group2(x)
        x = self.fc(x)

        return x

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


def resnet18(pretrained=False, model_root=None, **kwargs):
    model = Self_GroupingResNetModel(BasicBlock, [2, 2, 2, 2], **kwargs).cuda()
    if pretrained:
        misc.load_state_dict(model, model_urls['resnet18'], model_root)
    return model


def resnet34(pretrained=False, model_root=None, **kwargs):
    model = Self_GroupingResNetModel(BasicBlock, [3, 4, 6, 3], **kwargs).cuda()
    if pretrained:
        misc.load_state_dict(model, model_urls['resnet34'], model_root)
    return model


def resnet50(pretrained=False, model_root=None, **kwargs):
    model = Self_GroupingResNetModel(Bottleneck, [3, 4, 6, 3], **kwargs).cuda()
    if pretrained:
        misc.load_state_dict(model, model_urls['resnet50'], model_root)
    return model


def resnet101(pretrained=False, model_root=None, **kwargs):
    model = Self_GroupingResNetModel(Bottleneck, [3, 4, 23, 3], **kwargs).cuda()
    if pretrained:
        misc.load_state_dict(model, model_urls['resnet101'], model_root)
    return model


def resnet152(pretrained=False, model_root=None, **kwargs):
    model = Self_GroupingResNetModel(Bottleneck, [3, 8, 36, 3], **kwargs).cuda()
    if pretrained:
        misc.load_state_dict(model, model_urls['resnet152'], model_root)
    return model


class resnet(_fasterRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
    #self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
    #self.model_path = '/data/qingbeiguo/work/faster-rcnn.pytorch-pytorch-1.0/faster-rcnn.pytorch-pytorch-1.0/lib/model/faster_rcnn/pretrained_model/resnet101_caffe.pth'
    self.model_path = '/data/qingbeiguo/work/gcnn-Faster-rcnn/faster-rcnn.pytorch-pytorch-1.0-1-res50-coco-600-G60/lib/model/faster_rcnn/pretrained_model/model_training_G60.pth'
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    #resnet = resnet101()
    resnet = resnet50()
    print("resnet", resnet)

    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})
      print("resnet", resnet)

    # Build resnet.
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)
    print("self.RCNN_base", self.RCNN_base)

    self.RCNN_top = nn.Sequential(resnet.layer4)
    print("self.RCNN_top", self.RCNN_top)

    self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(2048, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

    # Fix blocks
    for p in self.RCNN_base[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base[1].parameters(): p.requires_grad=False

    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_base[6].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_base[5].parameters(): p.requires_grad=False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_base[4].parameters(): p.requires_grad=False

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

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7
