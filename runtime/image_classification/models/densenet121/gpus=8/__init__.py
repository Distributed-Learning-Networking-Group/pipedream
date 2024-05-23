# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

# import torchmodules.torchgraph as torchgraph
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
def arch():
    return "densenet121"

class Densenet121():
    def __init__(self, declares, calculations):
        self.declares = declares
        self.calculations = calculations

    def generate_layer_blocks(self):
        self.layers = {}
        for layer in self.declares.split('\n'):
            m = re.search(r'self.layer([0-9]+)', layer)
            # print(m)
            layer_id = int(m.group(1))
            self.layers[layer_id] = layer
        self.blocks = [[]]
        for line in self.calculations.split('\n'):
            self.blocks[-1].append(line)

            self.blocks.append([])
        # print(self.layers)
        # print("len blocks",len(self.blocks))
        # print(len(self.layers))
        # print(len(self.blocks))
        # print("end")

    def generate_stage(self, start, end):
        inputs = []
        outputs = []
        declare = []
        calculation = []
        for i in range(start, end):
            for line in self.blocks[i]:
                calculation.append(line)
                m = re.search(r'self.layer([0-9]+)', line)
                if m is not None:
                    layer_id = int(m.group(1))
                    declare.append(self.layers[layer_id])
                out = re.findall(r'out\d+', line)
                for arg in out[1:]:
                    if arg not in outputs and arg not in inputs:
                        inputs.append(arg)
                if out[0] not in outputs:
                    outputs.append(out[0])
        declare.append("self._initialize_weights()")
        return declare, calculation, inputs, outputs


class Stage(torch.nn.Module):
    def __init__(self, inputs, outputs, declares, calcus, fraction):
        super(Stage, self).__init__()
        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)
        exec('\n'.join(declares))
        back = int(fraction * len(calcus))
        if back == len(calcus):
            no_cp_ = ["{} = args[{}]".format(name, i)
                      for i, name in enumerate(inputs)]
            no_cp_.append("cp_out = cp.checkpoint(self.cp_forward, {}, self.dummy)".format(
                ','.join(inputs)))

            cp_ = calcus
            cp_i = 0
            cp_return = []
            no_cp_return = []
            for output in outputs:
                if output not in inputs:
                    cp_return.append(output)
                    no_cp_return.append("cp_out[{}]".format(cp_i))
                    cp_i += 1
                else:
                    no_cp_return.append(output)

            cp_ = ["{} = args[{}]".format(name, i)
                   for i, name in enumerate(inputs)] + cp_
            cp_.append("self.cp_out = ({},)".format(', '.join(cp_return)))
            no_cp_.append("self.out = ({},)".format(', '.join(no_cp_return)))

            self.cp = '\n'.join(cp_)
            self.no_cp = '\n'.join(no_cp_)
        elif back == 0:
            self.cp = "assert 1 == 0"
            no_cp_ = calcus

            no_cp_ = ["{} = args[{}]".format(name, i)
                      for i, name in enumerate(inputs)] + no_cp_
            no_cp_.append("self.out = ({})".format(', '.join(outputs)))

            self.no_cp = '\n'.join(no_cp_)
        else:
            no_cp_ = calcus[:-back]
            cp_ = calcus[-back:]

            no_cp_ = ["{} = args[{}]".format(name, i)
                      for i, name in enumerate(inputs)] + no_cp_

            cp_inputs = []
            cp_outputs = []
            for line in cp_:
                out = re.findall(r'out\d+', line)
                for arg in out[1:]:
                    if arg not in cp_outputs and arg not in cp_inputs:
                        cp_inputs.append(arg)
                if out[0] not in cp_outputs:
                    cp_outputs.append(out[0])

            cp_i = 0
            cp_return = []
            no_cp_return = []
            for output in outputs:
                if output in cp_outputs:
                    cp_return.append(output)
                    no_cp_return.append("cp_out[{}]".format(cp_i))
                    cp_i += 1
                else:
                    no_cp_return.append(output)

            no_cp_.append("cp_out = cp.checkpoint(self.cp_forward, {})".format(
                ', '.join(cp_inputs)))
            no_cp_.append("self.out = ({},)".format(', '.join(no_cp_return)))
            cp_ = ["{} = args[{}]".format(name, i)
                   for i, name in enumerate(cp_inputs)] + cp_
            cp_.append("self.cp_out = ({},)".format(', '.join(cp_return)))

            self.cp = '\n'.join(cp_)
            self.no_cp = '\n'.join(no_cp_)

    def forward(self, *args):
        exec(self.no_cp)
        return self.out

    def cp_forward(self, *args):
        exec(self.cp)
        return self.cp_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)


def replace(inputs):
    for i in range(len(inputs)):
        if inputs[i] == 'out0':
            inputs[i] = 'input0'
        # elif inputs[i] == 'out1':
        #     inputs[i] = 'input1'
        # elif inputs[i] == 'out2':
        #     inputs[i] = 'input2'
    return inputs


def model_densenet121(criterion, partition, recompute_ratio):
    _declares = get_declares()
    _calculations = get_caculations()
    module = Densenet121(_declares, _calculations)
    module.generate_layer_blocks()
    start = 0
    inputs = []
    outputs = [['out13']]
    all_outputs = []
    declares = []
    calculations = []
    for i in partition:
        stage = module.generate_stage(start, start + i)
        start += i
        declares.append(stage[0])
        calculations.append(stage[1])
        inputs.append(stage[2])
        all_outputs.append(stage[3])

    for i in range(len(partition) - 1, 0, -1):
        previous_output = []
        for name in inputs[i]:
            if name != 'out0':
                previous_output.append(name)
        for name in outputs[0]:
            if name not in all_outputs[i] and name not in previous_output:
                previous_output.append(name)
        outputs.insert(0, previous_output)

    ret = []
    for index in range(0, len(partition)):
        ret.append((
            Stage(inputs[index], outputs[index], declares[index], calculations[index], recompute_ratio[index]), replace(inputs[index]), outputs[index]))
    ret.append((criterion, outputs[len(partition)-1], ["loss"]))
    return ret

def get_declares():
    return '''self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),('norm0', nn.BatchNorm2d(64)),('relu0', nn.ReLU(inplace=True)),('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),]))
self.block1 = _DenseBlock(num_layers=6, num_input_features=64,bn_size=4, growth_rate=32, drop_rate=0)
self.trans1 = _Transition(num_input_features=256, num_output_features=256 // 2)
self.block2 = _DenseBlock(num_layers=12, num_input_features=128, bn_size=4, growth_rate=32, drop_rate=0)
self.trans2 = _Transition(num_input_features=512, num_output_features=512 // 2)
self.block3 = _DenseBlock(num_layers=24, num_input_features=256,bn_size=4, growth_rate=32, drop_rate=0)
self.trans3 = _Transition(num_input_features=1024, num_output_features=1024 // 2)
self.block4 = _DenseBlock(num_layers=16, num_input_features=512,bn_size=4, growth_rate=32, drop_rate=0)
self.norm5 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.relu = nn.ReLU(inplace=True)
self.avg_pool = nn.AvgPool2d(7, stride=1)
self.classifier = nn.Linear(1024, num_classes)'''

def get_caculations():
    return '''out2 = self.features(out0)
out3 = self.block1(out2)
out4 = self.trans1(out3)
out5 = self.block2(out4)
out6 = self.trans2(out5)
out7 = self.block3(out6)
out8 = self.trans3(out7)
out9 = self.block4(out8)
out10 = self.norm5(out9)
out11 = self.relu(out10)
out12 = self.avg_pool(out11).view(1, -1)
out13 = self.classifier(out12)'''

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
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
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=10):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        self.block1 = _DenseBlock(num_layers=6, num_input_features=64,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.trans1 = _Transition(num_input_features=256, num_output_features=256 // 2)
        self.block2 = _DenseBlock(num_layers=12, num_input_features=128, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.trans2 = _Transition(num_input_features=512, num_output_features=512 // 2)
        self.block3 = _DenseBlock(num_layers=24, num_input_features=256,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.trans3 = _Transition(num_input_features=1024, num_output_features=1024 // 2)
        self.block4 = _DenseBlock(num_layers=16, num_input_features=512,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.norm5 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # Linear layer
        num_features = 1024
        self.classifier = nn.Linear(num_features, num_classes)
        # ReLU layer
        self.relu = nn.ReLU(inplace=True)
        # Average pooling layer
        self.avg_pool = nn.AvgPool2d(7, stride=1)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        features = self.block1(features)
        print("finish block1")
        features = self.trans1(features)
        print("finish trans1")
        features = self.block2(features)
        print("finish block2")
        features = self.trans2(features)
        print("finish trans2")
        features = self.block3(features)
        print("finish block3")
        features = self.trans3(features)
        print("finish trans3")
        features = self.block4(features)
        print("finish block4")
        features = self.norm5(features)
        print("finish norm5",features.shape)
        out = self.relu(features)
        print("finish relu")
        out = self.avg_pool(out).view(1, -1)
        print("finish avg")
        out = self.classifier(out)
        return out
class DenseNet_orign(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=10):

        super(DenseNet_orign, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
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
        print("classifier",num_features)
        self.classifier = nn.Linear(num_features, num_classes)
        # ReLU layer
        self.relu = nn.ReLU(inplace=True)
        # Average pooling layer
        self.avg_pool = nn.AvgPool2d(7, stride=1)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        print(features.shape)
        out = self.relu(features)
        print(out.shape)
        out = self.avg_pool(out).view(features.size(0), -1)
        print(features.size(0))
        out = self.classifier(out)
        return out
# input_image = torch.randn(1, 3, 224, 224)

# densenet = DenseNet()
# output_tensor = densenet(input_image)
# print(output_tensor.shape)