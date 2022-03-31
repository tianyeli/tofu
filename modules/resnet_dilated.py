'''
resnet for segmentation or other fully conv purpose
from: https://github.com/warmspringwinds/pytorch-segmentation-detection
@ d6e7e82

'''

import torch.nn as nn
# import torchvision.models as models
import modules.resnet_orig as models

class Resnet101_8s(nn.Module):

    def __init__(self, num_classes=1000):

        super(Resnet101_8s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet101_8s = models.resnet101(fully_conv=True,
                                        pretrained=True,
                                        output_stride=8,
                                        remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet101_8s.fc = nn.Conv2d(resnet101_8s.inplanes, num_classes, 1)

        self.resnet101_8s = resnet101_8s

        self._normal_initialization(self.resnet101_8s.fc)


    def _normal_initialization(self, layer):

        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x):

        input_spatial_dim = x.size()[2:]

        x = self.resnet101_8s(x)

        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)

        return x



class Resnet18_8s(nn.Module):


    def __init__(self, num_classes=1000):

        super(Resnet18_8s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = models.resnet18(fully_conv=True,
                                      pretrained=True,
                                      output_stride=8,
                                      remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Conv2d(resnet18_8s.inplanes, num_classes, 1)

        self.resnet18_8s = resnet18_8s

        self._normal_initialization(self.resnet18_8s.fc)


    def _normal_initialization(self, layer):

        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x):

        input_spatial_dim = x.size()[2:]

        x = self.resnet18_8s(x)

        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)

        return x


class Resnet18_16s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet18_16s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 16
        resnet18_16s = models.resnet18(fully_conv=True,
                                      pretrained=True,
                                      output_stride=16,
                                      remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_16s.fc = nn.Conv2d(resnet18_16s.inplanes, num_classes, 1)
        
        self.resnet18_16s = resnet18_16s
        
        self._normal_initialization(self.resnet18_16s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet18_16s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x
    

class Resnet18_32s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet18_32s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_32s = models.resnet18(fully_conv=True,
                                       pretrained=True,
                                       output_stride=32,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_32s.fc = nn.Conv2d(resnet18_32s.inplanes, num_classes, 1)
        
        self.resnet18_32s = resnet18_32s
        
        self._normal_initialization(self.resnet18_32s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet18_32s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x
    

    
class Resnet34_32s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet34_32s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_32s = models.resnet34(fully_conv=True,
                                       pretrained=True,
                                       output_stride=32,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_32s.fc = nn.Conv2d(resnet34_32s.inplanes, num_classes, 1)
        
        self.resnet34_32s = resnet34_32s
        
        self._normal_initialization(self.resnet34_32s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet34_32s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x

    
class Resnet34_16s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet34_16s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_16s = models.resnet34(fully_conv=True,
                                       pretrained=True,
                                       output_stride=16,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_16s.fc = nn.Conv2d(resnet34_16s.inplanes, num_classes, 1)
        
        self.resnet34_16s = resnet34_16s
        
        self._normal_initialization(self.resnet34_16s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet34_16s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x


class Resnet34_8s(nn.Module):
    
    
    def __init__(self, num_classes=1000, pretrained=False, deformable_label=None):
        
        super(Resnet34_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_8s = models.resnet34(fully_conv=True,
                                       pretrained=pretrained,
                                       output_stride=8,
                                       remove_avg_pool_layer=True,
                                       deformable_label=deformable_label)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_8s.fc = nn.Conv2d(resnet34_8s.inplanes, num_classes, 1)
        
        self.resnet34_8s = resnet34_8s
        
        self._normal_initialization(self.resnet34_8s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet34_8s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x
    


class Resnet34_8s_skip(nn.Module):

    def __init__(self, num_classes=1000, pretrained=False):

        super(Resnet34_8s_skip, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_32s = models.resnet34( fully_conv=True,
                                        pretrained=pretrained,
                                        output_stride=32,
                                        remove_avg_pool_layer=True)
        # import ipdb; ipdb.set_trace()

        resnet_block_expansion_rate = resnet34_32s.layer1[0].expansion

        # Create a linear layer -- we don't need logits in this case
        resnet34_32s.fc = nn.Sequential()

        self.resnet34_32s = resnet34_32s

        self.score_32s = nn.Conv2d(512 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)

        self.score_16s = nn.Conv2d(256 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)

        self.score_8s = nn.Conv2d(128 *  resnet_block_expansion_rate,
                                   num_classes,
                                   kernel_size=1)


    def forward(self, x):

        input_spatial_dim = x.size()[2:]

        x = self.resnet34_32s.conv1(x)
        x = self.resnet34_32s.bn1(x)
        x = self.resnet34_32s.relu(x)
        x = self.resnet34_32s.maxpool(x)

        x = self.resnet34_32s.layer1(x)

        x = self.resnet34_32s.layer2(x)
        logits_8s = self.score_8s(x)

        x = self.resnet34_32s.layer3(x)
        logits_16s = self.score_16s(x)

        x = self.resnet34_32s.layer4(x)
        logits_32s = self.score_32s(x)

        logits_16s_spatial_dim = logits_16s.size()[2:]
        logits_8s_spatial_dim = logits_8s.size()[2:]

        logits_16s += nn.functional.upsample_bilinear(logits_32s,
                                        size=logits_16s_spatial_dim)

        logits_8s += nn.functional.upsample_bilinear(logits_16s,
                                        size=logits_8s_spatial_dim)

        logits_upsampled = nn.functional.upsample_bilinear(logits_8s,
                                                           size=input_spatial_dim)

        return logits_upsampled



class Resnet50_32s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet50_32s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_32s = models.resnet50(fully_conv=True,
                                       pretrained=True,
                                       output_stride=32,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_32s.fc = nn.Conv2d(resnet50_32s.inplanes, num_classes, 1)
        
        self.resnet50_32s = resnet50_32s
        
        self._normal_initialization(self.resnet50_32s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_32s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x

    
class Resnet50_16s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet50_16s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 16
        resnet50_8s = models.resnet50(fully_conv=True,
                                       pretrained=True,
                                       output_stride=16,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Conv2d(resnet50_8s.inplanes, num_classes, 1)
        
        self.resnet50_8s = resnet50_8s
        
        self._normal_initialization(self.resnet50_8s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_8s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x

class Resnet50_8s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet50_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_8s = models.resnet50(fully_conv=True,
                                       pretrained=True,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Conv2d(resnet50_8s.inplanes, num_classes, 1)
        
        self.resnet50_8s = resnet50_8s
        
        self._normal_initialization(self.resnet50_8s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_8s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x

    

class Resnet9_8s(nn.Module):
    
    # Gets ~ 46 MIOU on Pascal Voc
    
    def __init__(self, num_classes=1000):
        
        super(Resnet9_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = models.resnet18(fully_conv=True,
                                      pretrained=True,
                                      output_stride=8,
                                      remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Conv2d(resnet18_8s.inplanes, num_classes, 1)
        
        self.resnet18_8s = resnet18_8s
        
        self._normal_initialization(self.resnet18_8s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet18_8s.conv1(x)
        x = self.resnet18_8s.bn1(x)
        x = self.resnet18_8s.relu(x)
        x = self.resnet18_8s.maxpool(x)

        x = self.resnet18_8s.layer1[0](x)
        x = self.resnet18_8s.layer2[0](x)
        x = self.resnet18_8s.layer3[0](x)
        x = self.resnet18_8s.layer4[0](x)
        
        x = self.resnet18_8s.fc(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x