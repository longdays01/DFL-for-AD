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

class FADNet(nn.Module):
    def __init__(self):
        super(FADNet, self).__init__()
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

        self.conv2 = Conv2d(32, 32, (1, 1), stride=2)
        self.conv2_support = Conv2d(32, 256, (1, 1), stride=7)

        self.res_block2 = nn.Sequential(OrderedDict([
            ('batch_norm', nn.BatchNorm2d(32)),
            ('relu', nn.ReLU()),
            ('conv2d', Conv2d(32, 64, (3, 3), stride=2)),
            ('batch_norm_1', nn.BatchNorm2d(64)),
            ('relu_1', nn.ReLU()),
            ('conv2d_2', Conv2d(64, 64, (3, 3)))
        ]))
        self.conv3 = Conv2d(32, 64, (1, 1), stride=2)
        self.conv3_support = Conv2d(32, 256, (1, 1), stride=4)

        self.res_block3 = nn.Sequential(OrderedDict([
            ('batch_norm', nn.BatchNorm2d(64)),
            ('relu', nn.ReLU()),
            ('conv2d', Conv2d(64, 128, (3, 3), stride=2)),
            ('batch_norm_1', nn.BatchNorm2d(128)),
            ('relu_1', nn.ReLU()),
            ('conv2d_2', Conv2d(128, 128, (3, 3)))
        ]))
        self.conv4 = Conv2d(64, 128, (1, 1), stride=2)
        self.conv4_support = Conv2d(64, 256, (1, 1), stride=2)

        self.fc = nn.Linear(7, 7)
        self.dropout = nn.Dropout2d(p=0.5)
        self.relu = nn.ReLU()

        self.fc_accumulation = nn.Linear(3, 1)

    def forward(self, inputs):
        x1 = self.conv1(inputs)
        x1 = self.max_pool1(x1)

        
        x2 = self.res_block1(x1)
        x2 = torch.add(self.conv2(x1), x2)

        f1 = self.conv2_support(x1)
        f1 = f1.view(inputs.shape[0], -1).reshape(inputs.shape[0], FEATURE_SIZE, -1)
        
        f1 = f1.mean(axis=-1)

        
        x3 = self.res_block2(x2)
        x3 = torch.add(self.conv3(x2), x3)

        f2 = self.conv3_support(x2)
        f2 = f2.view(inputs.shape[0], -1).reshape(inputs.shape[0], FEATURE_SIZE, -1)
        
        f2 = f2.mean(axis=-1)

        
        x4 = self.res_block3(x3)
        x4 = torch.add(self.conv4(x3), x4)

        f3 = self.conv4_support(x3)
        f3 = f3.view(inputs.shape[0], -1).reshape(inputs.shape[0], FEATURE_SIZE, -1)
        
        f3 = f3.mean(axis=-1)

        
        x4 = self.fc(x4)
        x4 = self.relu(x4)
        x4 = self.dropout(x4)
        x4 = x4.view(inputs.shape[0], -1)

        
        f_feature = self.fc_accumulation(torch.stack([f1, f2, f3], axis=2)).squeeze(-1)

        
        x_final = torch.mul(x4, f_feature)
        return x_final.mean(axis=1).unsqueeze(1)

class DrivingNet(Model):
    def __init__(self, model, criterion, metric, device,
                 optimizer_name="adam", lr_scheduler="sqrt", initial_lr=1e-3, epoch_size=1):
        super(DrivingNet, self).__init__()
        if model == "FADNet":
            self.net = FADNet().to(device)
        elif model == "FADNet_plus":
            self.net = FADNet_plus().to(device)
        else: self.net = ADTVNet().to(device)
    
        self.criterion = criterion
        self.metric = metric
        self.device = device

        self.optimizer = get_optimizer(optimizer_name, self.net, initial_lr)
        self.lr_scheduler = get_lr_scheduler(self.optimizer, lr_scheduler, epoch_size)

    def fit_iterator_one_epoch(self, iterator):
        epoch_loss = 0
        epoch_acc = 0

        self.net.train()

        for x, y in iterator:
            self.optimizer.zero_grad()
            x = x.to(self.device, dtype=torch.float)
            y = y.to(self.device, dtype=torch.float).unsqueeze(-1)
            predictions = self.net(x)

            loss = self.criterion(predictions, y)

            acc = self.metric(predictions, y)

            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def fit_batch(self, iterator, update=True):
        self.net.train()

        x, y = next(iter(iterator))

        x = x.to(self.device, dtype=torch.float)
        y = y.to(self.device, dtype=torch.float).unsqueeze(-1)

        self.optimizer.zero_grad()

        predictions = self.net(x)

        loss = self.criterion(predictions, y)

        acc = self.metric[0](predictions, y)

        loss.backward()

        if update:
            self.optimizer.step()
            

        batch_loss = loss.item()
        batch_acc = acc.item()

        return batch_loss, batch_acc

    def evaluate_iterator(self, iterator):
        epoch_loss = 0
        epoch_acc = 0

        self.net.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm.tqdm(iterator)):
                x = x.to(self.device)
                y = y.unsqueeze(-1).to(self.device)
                predictions = self.net(x)

                loss = self.criterion(predictions, y)

                acc = self.metric[0](predictions, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)











