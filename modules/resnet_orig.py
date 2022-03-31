'''
from: https://github.com/warmspringwinds/vision/blob/eb6c13d3972662c55e752ce7a376ab26a1546fb5/torchvision/models/resnet.py
a particular version to make resnet_dilated.py work

ref: https://github.com/warmspringwinds/pytorch-segmentation-detection/issues/20

'''

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1, deformable=False):
    "3x3 convolution with padding"
    
    kernel_size = np.asarray((3, 3))
    
    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
    
    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2
    
    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    if deformable:
        from modules.deformable_conv.modules.deform_conv import DeformConvPack
        return DeformConvPack(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                         padding=full_padding, dilation=dilation, bias=False, lr_mult=0.1,
                         groups=1, deformable_groups=1).cuda()
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                         padding=full_padding, dilation=dilation, bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, deformable=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation, deformable=deformable)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation, deformable=deformable)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    # note: not yet updated for deformable convolution
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = conv3x3(planes, planes, stride=stride, dilation=dilation)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 fully_conv=False,
                 remove_avg_pool_layer=False,
                 output_stride=32,
                 additional_blocks=0,
                 multi_grid=(1,1,1),
                 deformable_label=None ):
        
        # Add additional variables to track
        # output stride. Necessary to achieve
        # specified output stride.
        self.output_stride = output_stride
        self.current_stride = 4
        self.current_dilation = 1
        
        self.remove_avg_pool_layer = remove_avg_pool_layer
        
        self.inplanes = 64
        self.fully_conv = fully_conv
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # expect: e.g. deformable_label = [ [0] * 3, [0] * 4, [0] * 6, [1,1,1] ] and corresponding layers = [3,4,6,3]
        if deformable_label is not None:
            for idx, this_layer_num in enumerate(layers):
                if len(deformable_label[idx]) != this_layer_num:
                    raise RuntimeError( 'deformable_labels does not match to layer nums' )
        if deformable_label is None:
            dls = [None] * len(layers)
            self.deformable = False
        else:
            dls = deformable_label
            self.deformable = True

        self.layer1 = self._make_layer(block, 64, layers[0], dl=dls[0])
        self.layer2 = self._make_layer(block, 128, layers[1], dl=dls[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], dl=dls[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], dl=dls[3], stride=2, multi_grid=multi_grid)
        
        self.additional_blocks = additional_blocks

        # not updated deformable for additional_blocks
        if additional_blocks == 1:

            self.layer5 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)

        if additional_blocks == 2:

            self.layer5 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)
            self.layer6 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)

        if additional_blocks == 3:

            self.layer5 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)
            self.layer6 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)
            self.layer7 = self._make_layer(block, 512, layers[3], stride=2, multi_grid=multi_grid)

        self.avgpool = nn.AvgPool2d(7)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.fully_conv:
            self.avgpool = nn.AvgPool2d(7, padding=3, stride=1)
            # In the latest unstable torch 4.0 the tensor.copy_
            # method was changed and doesn't work as it used to be
            #self.fc = nn.Conv2d(512 * block.expansion, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    
    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    multi_grid=None,
                    dl=None):
        
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            # Check if we already achieved desired output stride.
            if self.current_stride == self.output_stride:
                # If so, replace subsampling with a dilation to preserve
                # current spatial resolution.
                self.current_dilation = self.current_dilation * stride
                stride = 1
            else:
                # If not, perform subsampling and update current
                # new output stride.
                self.current_stride = self.current_stride * stride
                
            # We don't dilate 1x1 convolution.
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        
        # layer 0
        dilation = multi_grid[0] * self.current_dilation if multi_grid else self.current_dilation
        if dl is None or dl[0] == 0:
            layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation, deformable=True))

        self.inplanes = planes * block.expansion
        
        # the following layers
        for i in range(1, blocks):
            
            dilation = multi_grid[i] * self.current_dilation if multi_grid else self.current_dilation

            if dl is None:
                this_block = block(self.inplanes, planes, dilation=dilation)
            elif dl[i] == 0:
                this_block = block(self.inplanes, planes, dilation=dilation)
            elif dl[i] == 1:
                this_block = block(self.inplanes, planes, dilation=dilation, deformable=True)

            layers.append(this_block)

        return nn.Sequential(*layers)

    def load_pretrained(self, model_path=None):
        # note: currently rely on outside module to correctly load the pretrained parameters
        raise NotImplementedError

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.additional_blocks == 1:
            
            x = self.layer5(x)
        
        if self.additional_blocks == 2:
            
            x = self.layer5(x)
            x = self.layer6(x)
        
        if self.additional_blocks == 3:
            
            x = self.layer5(x)
            x = self.layer6(x)
            x = self.layer7(x)
        
        if not self.remove_avg_pool_layer:
            x = self.avgpool(x)
        
        if not self.fully_conv:
            x = x.view(x.size(0), -1)
            
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    
    
    if pretrained:
        
        if model.additional_blocks:
            
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
            
            return model
           
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    layers = [3, 4, 6, 3]
    model = ResNet(BasicBlock, layers, **kwargs)

    if pretrained:
        if model.additional_blocks:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
            return model
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        print('loaded pretrained')
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    
    if pretrained:
        
        if model.additional_blocks:
            
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
            
            return model
           
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    
   
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    
    
    if pretrained:
        
        if model.additional_blocks:
            
            model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
            
            return model
           
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    
   
    return model
    


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
