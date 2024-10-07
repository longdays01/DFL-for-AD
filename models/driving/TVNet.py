import math
from collections import OrderedDict
import torch.nn as nn
import torch
from utils.optim import get_optimizer, get_lr_scheduler
from ..model import Model
from torch.nn import functional as F
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
import torchvision.models as models
from torchvision.models.inception import InceptionOutputs
import timm
from transformers import MobileViTModel
import torch.nn.init as init  
from transformers import AutoImageProcessor, SwinForImageClassification

NUMBER_CLASSES = 1
FEATURE_SIZE = 6272
import tqdm
def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):

    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_needed = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows -
                  input_rows)
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)


class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


cuda0 = torch.device('cuda:0')
class FADNet_plus(nn.Module):
    def __init__(self):
        super(FADNet_plus, self).__init__()
        self.conv1 = Conv2d(1, 32, (5, 5), stride=2)
        self.max_pool1 = nn.MaxPool2d((3, 3), 2)
        self.res_block1 = nn.Sequential(OrderedDict([
            ('batch_norm', nn.BatchNorm2d(32)),
            ('relu', nn.ReLU()),
            ('conv2d', Conv2d(32, 32, (3, 3), stride=2)),
            ('batch_norm_1', nn.BatchNorm2d(32)),
            ('relu_1', nn.ReLU()),
            ('conv2d_2', Conv2d(32, 32, (3, 3)))
        ]))
        self.conv2 = Conv2d(32, 256, (1, 1), stride=7)

        self.res_block2 = nn.Sequential(OrderedDict([
            ('batch_norm', nn.BatchNorm2d(32)),
            ('relu', nn.ReLU()),
            ('conv2d', Conv2d(32, 64, (3, 3), stride=2)),
            ('batch_norm_1', nn.BatchNorm2d(64)),
            ('relu_1', nn.ReLU()),
            ('conv2d_2', Conv2d(64, 64, (3, 3)))
        ]))
        self.conv3 = Conv2d(32, 256, (1, 1), stride=4)

        self.res_block3 = nn.Sequential(OrderedDict([
            ('batch_norm', nn.BatchNorm2d(64)),
            ('relu', nn.ReLU()),
            ('conv2d', Conv2d(64, 128, (3, 3), stride=2)),
            ('batch_norm_1', nn.BatchNorm2d(128)),
            ('relu_1', nn.ReLU()),
            ('conv2d_2', Conv2d(128, 128, (3, 3)))
        ]))
        self.conv4 = Conv2d(64, 256, (1, 1), stride=2)
        self.dropout = nn.Dropout2d(p=0.5)
        self.relu = nn.ReLU()

        self.fc_feature = nn.Linear(3, 1)
        self.fc = nn.Linear(FEATURE_SIZE, NUMBER_CLASSES)

    def forward(self, inputs):
        x1 = inputs
        x1 = self.conv1(x1)
        x1 = self.max_pool1(x1)

        x2 = self.res_block1(x1)

        f1 = self.conv2(x1)
        f1 = f1.view(inputs.shape[0], -1).reshape(inputs.shape[0], FEATURE_SIZE, -1)
        
        f1 = f1.mean(axis=-1)

        x3 = self.res_block2(x2)

        f2 = self.conv3(x2)
        f2 = f2.view(inputs.shape[0], -1).reshape(inputs.shape[0], FEATURE_SIZE, -1)
        
        f2 = f2.mean(axis=-1)

        x4 = self.res_block3(x3)

        f3 = self.conv4(x3)
        f3 = f3.view(inputs.shape[0], -1).reshape(inputs.shape[0], FEATURE_SIZE, -1)
        
        f3 = f3.mean(axis=-1)

        x4 = self.relu(x4)
        x4 = self.dropout(x4)
        x4 = x4.view(inputs.shape[0], -1)

        
        x_feature = self.fc_feature(torch.stack([f1, f2, f3], axis=2)).squeeze(-1)

        
        x_final = torch.mul(x4, x_feature)
        
        return self.fc(x_final)

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('batch_norm', nn.BatchNorm2d(64)),
            ('relu', nn.ReLU()),
            ('max_pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        
        self.res_block1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('batch_norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('batch_norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU())
        ]))

        
        self.res_block2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)),
            ('batch_norm1', nn.BatchNorm2d(128)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)),
            ('batch_norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU())
        ]))

        
        self.res_block3 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)),
            ('batch_norm1', nn.BatchNorm2d(256)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)),
            ('batch_norm2', nn.BatchNorm2d(256)),
            ('relu2', nn.ReLU())
        ]))

        
        self.res_block4 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)),
            ('batch_norm1', nn.BatchNorm2d(512)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)),
            ('batch_norm2', nn.BatchNorm2d(512)),
            ('relu2', nn.ReLU())
        ]))

        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.5)

        
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        
        x = self.conv1(x)

        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        
        x = self.fc(x)

        return x

class ResNet8(nn.Module):
    def __init__(self):
        super(ResNet8, self).__init__()

        
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False)),
            ('batch_norm', nn.BatchNorm2d(32)),
            ('relu', nn.ReLU()),
            ('max_pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        
        self.res_block1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)),
            ('batch_norm1', nn.BatchNorm2d(32)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)),
            ('batch_norm2', nn.BatchNorm2d(32)),
            ('relu2', nn.ReLU())
        ]))

        
        self.res_block2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)),
            ('batch_norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('batch_norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU())
        ]))

        
        self.res_block3 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)),
            ('batch_norm1', nn.BatchNorm2d(128)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)),
            ('batch_norm2', nn.BatchNorm2d(128)),
            ('relu2', nn.ReLU())
        ]))

        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.5)

        
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        
        x = self.conv1(x)

        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        
        x = self.fc(x)

        return x

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out

class ADTVNet(nn.Module):
    def __init__(self):
        super(ADTVNet, self).__init__()
        self.conv1 = Conv2d(1, 32, (5, 5), stride=2)
        self.max_pool1 = nn.MaxPool2d((3, 3), 2)
        self.res_block1 = nn.Sequential(OrderedDict([
            ('batch_norm', nn.BatchNorm2d(32)),
            ('relu', nn.ReLU()),
            ('conv2d', Conv2d(32, 32, (3, 3), stride=2)),
            ('batch_norm_1', nn.BatchNorm2d(32)),
            ('relu_1', nn.ReLU()),
            ('conv2d_2', Conv2d(32, 32, (3, 3)))
        ]))
        self.conv2 = Conv2d(32, 256, (1, 1), stride=7)
        self.cbam1 = CBAM(gate_channels=256)

        self.res_block2 = nn.Sequential(OrderedDict([
            ('batch_norm', nn.BatchNorm2d(32)),
            ('relu', nn.ReLU()),
            ('conv2d', Conv2d(32, 64, (3, 3), stride=2)),
            ('batch_norm_1', nn.BatchNorm2d(64)),
            ('relu_1', nn.ReLU()),
            ('conv2d_2', Conv2d(64, 64, (3, 3)))
        ]))
        self.conv3 = Conv2d(32, 256, (1, 1), stride=4)
        self.cbam2 = CBAM(gate_channels=256)

        self.res_block3 = nn.Sequential(OrderedDict([
            ('batch_norm', nn.BatchNorm2d(64)),
            ('relu', nn.ReLU()),
            ('conv2d', Conv2d(64, 128, (3, 3), stride=2)),
            ('batch_norm_1', nn.BatchNorm2d(128)),
            ('relu_1', nn.ReLU()),
            ('conv2d_2', Conv2d(128, 128, (3, 3)))
        ]))
        self.conv4 = Conv2d(64, 256, (1, 1), stride=2)
        self.cbam3 = CBAM(gate_channels=256)

        self.dropout = nn.Dropout2d(p=0.5)
        self.relu = nn.ReLU()

        self.fc_feature = nn.Linear(3, 1)
        self.fc = nn.Linear(6272, 1)

    def forward(self, inputs):
        x1 = inputs
        x1 = self.conv1(x1)
        x1 = self.max_pool1(x1)

        x2 = self.res_block1(x1)
        f1 = self.cbam1(self.conv2(x1))
        f1 = f1.view(inputs.shape[0], -1).reshape(inputs.shape[0], 6272, -1)
        f1 = f1.mean(axis=-1)

        x3 = self.res_block2(x2)
        f2 = self.cbam2(self.conv3(x2))
        f2 = f2.view(inputs.shape[0], -1).reshape(inputs.shape[0], 6272, -1)
        f2 = f2.mean(axis=-1)

        x4 = self.res_block3(x3)
        f3 = self.cbam3(self.conv4(x3))
        f3 = f3.view(inputs.shape[0], -1).reshape(inputs.shape[0], 6272, -1)
        f3 = f3.mean(axis=-1)

        x4 = self.relu(x4)
        x4 = self.dropout(x4)
        x4 = x4.view(inputs.shape[0], -1)

        x_feature = self.fc_feature(torch.stack([f1, f2, f3], axis=2)).squeeze(-1)
        x_final = torch.mul(x4, x_feature)

        return self.fc(x_final)



        
class ResBlockCBAM(nn.Module):
    def __init__(self, res_block, in_channels, out_channels):
        super(ResBlockCBAM, self).__init__()
        self.res_block = res_block
        self.cbam = CBAM(gate_channels=out_channels)
        
        
        self.conv1x1 = None
        if in_channels != out_channels:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)

    def forward(self, x):
        residual = x
        
        
        out = self.res_block(x)
        out = self.cbam(out)
        
        
        if self.conv1x1 is not None:
            residual = self.conv1x1(residual)

        
        out += residual
        return out
    

class AttentionADTVNet(nn.Module):
    def __init__(self):
        super(AttentionADTVNet, self).__init__()
        
        
        resnet = models.resnet18(pretrained=True)
        
        
        self.initial_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        
        self.res_block1 = ResBlockCBAM(resnet.layer1, 64, 64)
        self.res_block2 = ResBlockCBAM(resnet.layer2, 64, 128)
        self.res_block3 = ResBlockCBAM(resnet.layer3, 128, 256)
        self.res_block4 = ResBlockCBAM(resnet.layer4, 256, 512)

        
        self.fc1 = nn.Linear(512, 256)  
        self.fc2 = nn.Linear(256, 1)    

    def forward(self, x):
        
        x = self.initial_layers(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        
        x = torch.mean(x, dim=[2, 3])
        
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class InceptionNet(nn.Module):
    def __init__(self):
        super(InceptionNet, self).__init__()
        self.model = models.inception_v3(pretrained=True)

        
        self.model.Conv2d_1a_3x3.conv = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

        
        self.model.fc = nn.Linear(self.model.fc.in_features, NUMBER_CLASSES)

    def forward(self, x):
        
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        
        x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        return self.model(x)

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        
        self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, NUMBER_CLASSES)

    def forward(self, x):
        
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(x)

class VGG16(nn.Module):
    def __init__(self, pretrained=True, freeze_layers=True):
        super(VGG16, self).__init__()
        self.model = models.vgg16(pretrained=pretrained)

        
        if freeze_layers:
            for param in self.model.features.parameters():
                param.requires_grad = False

        
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, NUMBER_CLASSES)

    def forward(self, x):
        
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        return self.model(x)

class DAVE2(nn.Module):
    def __init__(self):
        super(DAVE2, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),  
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2), 
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2), 
            nn.ELU(),
            nn.Conv2d(48, 64, 3),           
            nn.ELU(),
            nn.Conv2d(64, 64, 3),           
            nn.Dropout(0.5)
        )
        
        
        self.calculate_feature_size()

        self.dense_layers = nn.Sequential(
            nn.Linear(self.feature_size, 100),
            nn.ELU(),
            
            
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1)
        )

    def calculate_feature_size(self):
        
        dummy_input = torch.zeros(1, 3, 200, 200)  
        output = self.conv_layers(dummy_input)
        self.feature_size = output.view(1, -1).size(1)

    def forward(self, data):
        
        if data.size(1) == 1:
            data = data.repeat(1, 3, 1, 1)

        output = self.conv_layers(data)
        output = output.view(output.size(0), -1)
        output = self.dense_layers(output)
        return output
        
class DAVE2_base(nn.Module):
    def __init__(self):
        super(DAVE2_base, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),  
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2), 
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2), 
            nn.ELU(),
            nn.Conv2d(48, 64, 3),           
            nn.ELU(),
            nn.Conv2d(64, 64, 3),           
            nn.Dropout(0.5)
        )
        
        
        self.calculate_feature_size()

        self.dense_layers = nn.Sequential(
            nn.Linear(self.feature_size, 1164),
            nn.ELU(),
            nn.Linear(1164, 100),
            nn.ELU(),            
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1)
        )

    def calculate_feature_size(self):
        
        dummy_input = torch.zeros(1, 3, 200, 200)  
        output = self.conv_layers(dummy_input)
        self.feature_size = output.view(1, -1).size(1)

    def forward(self, data):
        
        if data.size(1) == 1:
            data = data.repeat(1, 3, 1, 1)

        output = self.conv_layers(data)
        output = output.view(output.size(0), -1)
        output = self.dense_layers(output)
        return output        

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet81(nn.Module):
    def __init__(self):
        super(ResNet81, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.res1 = BasicBlock(32, 32, stride=2)
        self.res2 = BasicBlock(32, 64, stride=2)
        self.res3 = BasicBlock(64, 128, stride=2)
        
        
        self.calculate_feature_size()

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1)
        )
    
    def calculate_feature_size(self):
        
        dummy_input = torch.zeros(1, 3, 200, 200)  
        output = self.conv1(dummy_input)
        output = self.pool1(output)
        output = self.res1(output)
        output = self.res2(output)
        output = self.res3(output)
        self.feature_size = output.view(1, -1).size(1)

    def forward(self, data):
        
        if data.size(1) == 1:
            data = data.repeat(1, 3, 1, 1)
        
        x = self.conv1(data)
        x = self.pool1(x)
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        x = self.flatten(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        output = self.fc(x)
        return output

class AttDAVE2(nn.Module):
    def __init__(self):
        super(AttDAVE2, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),  
            nn.ELU(),
            CBAM(24),
            nn.Conv2d(24, 36, 5, stride=2), 
            nn.ELU(),
            CBAM(36),
            nn.Conv2d(36, 48, 5, stride=2), 
            nn.ELU(),
            CBAM(48),
            nn.Conv2d(48, 64, 3),           
            nn.ELU(),
            CBAM(64),
            nn.Conv2d(64, 64, 3),           
            nn.Dropout(0.5)
        )
        
        
        self.calculate_feature_size()

        self.dense_layers = nn.Sequential(
            nn.Linear(self.feature_size, 100),  
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1)
        )

    def calculate_feature_size(self):
        
        dummy_input = torch.zeros(1, 3, 200, 200)  
        output = self.conv_layers(dummy_input)
        self.feature_size = output.view(1, -1).size(1)  

    def forward(self, data):
        
        if data.size(1) == 1:
            data = data.repeat(1, 3, 1, 1)

        output = self.conv_layers(data)
        output = output.view(output.size(0), -1)
        output = self.dense_layers(output)
        return output

class RandomNet(nn.Module):
    def __init__(self):
        super(RandomNet, self).__init__()
        
        self.dummy_param = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x):
        batch_size = x.size(0)
        
        random_output = torch.rand(batch_size, NUMBER_CLASSES, device=x.device)
        return random_output + self.dummy_param

class ConstantNet(nn.Module):
    def __init__(self, constant=0):
        super(ConstantNet, self).__init__()
        self.constant = constant
        
        self.dummy_param = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x):
        batch_size = x.size(0)
        
        return torch.full((batch_size, NUMBER_CLASSES), self.constant).to(x.device) + self.dummy_param

def calculate_feature_size(model, input_shape=(1, 1, 224, 224)):
    dummy_input = torch.zeros(*input_shape)
    output = model(dummy_input)
    feature_size = output.view(1, -1).size(1)
    return feature_size

class EfficientNetV2(nn.Module):
    def __init__(self):
        super(EfficientNetV2, self).__init__()
        self.model = timm.create_model('efficientnetv2_rw_s', pretrained=True)
        self.model.classifier = nn.Identity()  
        self.feature_size = self.calculate_feature_size(self.forward_features_only)
        self.fc = nn.Linear(self.feature_size, NUMBER_CLASSES)

    def forward_features_only(self, x):
        x = self.model.forward_features(x)
        return x

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.forward_features_only(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
        
    def calculate_feature_size(self, model):
        
        dummy_input = torch.zeros(1, 3, 224, 224)  
        output = model(dummy_input)
        return output.view(1, -1).size(1)  
        
    def reset_weights(self):
        for layer in self.model.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.GroupNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if hasattr(sublayer, 'reset_parameters'):
                        sublayer.reset_parameters()
    
        
        self.fc.reset_parameters()



class TinyViT(nn.Module):
    def __init__(self):
        super(TinyViT, self).__init__()
        
        self.model = timm.create_model('tinyvit_5m_224', pretrained=True)
        self.model.head = nn.Identity()  
        
        
        self.feature_size = calculate_feature_size(self.forward_features_only)
        
        
        self.fc = nn.Linear(self.feature_size, NUMBER_CLASSES)

    def forward_features_only(self, x):
        
        return self.model.forward_features(x)

    def forward(self, x):
        
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        
        x = self.forward_features_only(x)
        
        
        x = x.view(x.size(0), -1)
        
        
        return self.fc(x)


class MobileViT(nn.Module):
    def __init__(self):
        super(MobileViT, self).__init__()
        
        try:
            self.model = MobileViTModel.from_pretrained("apple/mobilevit-small")
        except Exception as e:
            print(f"Pretrained model loading failed: {e}. Initializing from scratch.")
            self.model = MobileViTModel(config=get_configuration())
            self.reset_weights()  

        
        self.model.pooler = nn.Identity()

        
        self.feature_size = self.calculate_feature_size(self.forward_features_only)

        
        self.fc = nn.Linear(self.feature_size, NUMBER_CLASSES)

    def forward_features_only(self, x):
        
        outputs = self.model(pixel_values=x)
        return outputs.last_hidden_state  

    def calculate_feature_size(self, model):
        
        dummy_input = torch.zeros(1, 3, 224, 224)  
        output = model(dummy_input)
        return output.view(1, -1).size(1)  

    def reset_weights(self):
        
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Embedding):
                nn.init.normal_(layer.weight, mean=0, std=1)
            elif isinstance(layer, nn.Parameter):
                nn.init.normal_(layer, mean=0, std=1)

        
        self.fc.reset_parameters()

    def forward(self, x):
        
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        
        x = self.forward_features_only(x)
        x = x.view(x.size(0), -1)  
        return self.fc(x)

class SwinTransformer(nn.Module):
    def __init__(self):
        super(SwinTransformer, self).__init__()
        
        
        self.model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

        
        self.model.classifier = nn.Identity()

        
        self.feature_size = self.calculate_feature_size()

        
        self.fc = nn.Linear(self.feature_size, 1)  

    def calculate_feature_size(self):
        
        dummy_input = torch.zeros(1, 3, 224, 224)  
        with torch.no_grad():
            outputs = self.model(dummy_input)
        return outputs.logits.view(1, -1).size(1)

    def forward(self, x):
        
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        
        

        
        outputs = self.model(x)
        x = outputs.logits

        
        x = x.view(x.size(0), -1)

        
        return self.fc(x)

class SwinTransformer_base(nn.Module):
    def __init__(self):
        super(SwinTransformer_base, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.model.head = nn.Identity()  
        self.feature_size = self.calculate_feature_size()
        self.fc = nn.Linear(self.feature_size, NUMBER_CLASSES)

    def calculate_feature_size(self):
        
        dummy_input = torch.zeros(1, 3, 224, 224)  
        dummy_output = self.model.forward_features(dummy_input)
        return dummy_output.view(1, -1).size(1)

    def forward(self, x):
        
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.model.forward_features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DrivingNet(Model):
    def __init__(self, model, criterion, metrics, device, optimizer_name="radam", lr_scheduler="sqrt", initial_lr=1e-3, epoch_size=1):
        super(DrivingNet, self).__init__()
        if model == "FADNet_plus":
            self.net = FADNet_plus().to(device)
        elif model == "SwinTransformer":
            self.net = SwinTransformer().to(device)
        elif model == "InceptionNet":
            self.net = InceptionNet().to(device)
        elif model == "MobileNet":
            self.net = MobileNet().to(device)
        elif model == "SwinTransformer_base":
            self.net = SwinTransformer_base().to(device)
        elif model == "RandomNet":
            self.net = RandomNet().to(device)
        elif model == "ConstantNet":
            self.net = ConstantNet().to(device)
        elif model == "DAVE2":
            self.net = DAVE2().to(device)
        elif model == "DAVE2_base":
            self.net = DAVE2_base().to(device)
        elif model == "ResNet8":
            self.net = ResNet8().to(device)
        elif model == "ResNet18":
            self.net = ResNet18().to(device)
        elif model == "EfficientNetV2":
            effnet_model = EfficientNetV2() 
            effnet_model.reset_weights() 
            self.net = effnet_model.to(device)      
        elif model == "MobileViT":
            mobilevit_model = MobileViT() 
            mobilevit_model.reset_weights()  
            self.net = mobilevit_model.to(device)  
        elif model == "TinyViT":
            self.net = TinyViT().to(device)
        else:
            self.net = ADTVNet().to(device)
    
        self.criterion = criterion
        self.metrics = metrics  
        self.device = device

        self.optimizer = get_optimizer(optimizer_name, self.net, initial_lr)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, lr_scheduler, epoch_size)

    def fit_iterator_one_epoch(self, iterator):
        epoch_loss = 0
        epoch_metrics = [0] * len(self.metrics)  

        self.net.train()

        for x, y in iterator:
            self.optimizer.zero_grad()
            x = x.to(self.device, dtype=torch.float)
            y = y.to(self.device, dtype=torch.float).unsqueeze(-1)
            predictions = self.net(x)

            loss = self.criterion(predictions, y)

            
            for idx, metric in enumerate(self.metrics):
                epoch_metrics[idx] += metric(predictions, y).item()

            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(iterator)
        avg_metrics = [metric / len(iterator) for metric in epoch_metrics]
        return avg_loss, avg_metrics

    def fit_batch(self, iterator, update=True):
        self.net.train()

        x, y = next(iter(iterator))

        x = x.to(self.device, dtype=torch.float)
        y = y.to(self.device, dtype=torch.float).unsqueeze(-1)

        self.optimizer.zero_grad()

        predictions = self.net(x)

        loss = self.criterion(predictions, y)
        
        
        batch_metrics = [metric(predictions, y).item() for metric in self.metrics]

        loss.backward()

        if update:
            self.optimizer.step()
            

        batch_loss = loss.item()
        
        
        batch_gradients = [
            param.grad.clone().detach() if param.grad is not None else torch.zeros_like(param)
            for param in self.net.parameters()
        ]

        return batch_loss, batch_metrics, batch_gradients

    def evaluate_iterator(self, iterator):
        epoch_loss = 0
        epoch_rmse = 0
        epoch_mae = 0  

        self.net.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm.tqdm(iterator)):
                x = x.to(self.device)
                y = y.unsqueeze(-1).to(self.device)
                predictions = self.net(x)

                loss = self.criterion(predictions, y)

                
                rmse = self.metrics[0](predictions, y)
                mae = self.metrics[1](predictions, y)

                epoch_loss += loss.item()
                epoch_rmse += rmse.item()
                epoch_mae += mae.item()  

        return epoch_loss / len(iterator), epoch_rmse / len(iterator), epoch_mae / len(iterator)  

    
    
    

    
    
    
    

    

    
    

    
    

    
    
    
    

    









