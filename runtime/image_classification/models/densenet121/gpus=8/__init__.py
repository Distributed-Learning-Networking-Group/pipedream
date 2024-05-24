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
        print(self.no_cp)
        print()
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
    outputs = [['out69']]
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
    return '''self.layer1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
self.layer2 = nn.BatchNorm2d(64)
self.layer3 = nn.ReLU(inplace=True)
self.layer4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
self.layer5 = _DenseLayer(64, 32, 4, 0)
self.layer6 = _DenseLayer(96, 32, 4, 0)
self.layer7 = _DenseLayer(128, 32, 4, 0)
self.layer8 = _DenseLayer(160, 32, 4, 0)
self.layer9 = _DenseLayer(192, 32, 4, 0)
self.layer10 = _DenseLayer(224, 32, 4, 0)
self.layer11 = _Transition(num_input_features=256, num_output_features=256 // 2)
self.layer12 = _DenseLayer(128, 32, 4, 0)
self.layer13 = _DenseLayer(160, 32, 4, 0)
self.layer14 = _DenseLayer(192, 32, 4, 0)
self.layer15 = _DenseLayer(224, 32, 4, 0)
self.layer16 = _DenseLayer(256, 32, 4, 0)
self.layer17 = _DenseLayer(288, 32, 4, 0)
self.layer18 = _DenseLayer(320, 32, 4, 0)
self.layer19 = _DenseLayer(352, 32, 4, 0)
self.layer20 = _DenseLayer(384, 32, 4, 0)
self.layer21 = _DenseLayer(416, 32, 4, 0)
self.layer22 = _DenseLayer(448, 32, 4, 0)
self.layer23 = _DenseLayer(480, 32, 4, 0)
self.layer24 = _Transition(num_input_features=512, num_output_features=512 // 2)
self.layer25 = _DenseLayer(256, 32, 4, 0)
self.layer26 = _DenseLayer(288, 32, 4, 0)
self.layer27 = _DenseLayer(320, 32, 4, 0)
self.layer28 = _DenseLayer(352, 32, 4, 0)
self.layer29 = _DenseLayer(384, 32, 4, 0)
self.layer30 = _DenseLayer(416, 32, 4, 0)
self.layer31 = _DenseLayer(448, 32, 4, 0)
self.layer32 = _DenseLayer(480, 32, 4, 0)
self.layer33 = _DenseLayer(512, 32, 4, 0)
self.layer34 = _DenseLayer(544, 32, 4, 0)
self.layer35 = _DenseLayer(576, 32, 4, 0)
self.layer36 = _DenseLayer(608, 32, 4, 0)
self.layer37 = _DenseLayer(640, 32, 4, 0)
self.layer38 = _DenseLayer(672, 32, 4, 0)
self.layer39 = _DenseLayer(704, 32, 4, 0)
self.layer40 = _DenseLayer(736, 32, 4, 0)
self.layer41 = _DenseLayer(768, 32, 4, 0)
self.layer42 = _DenseLayer(800, 32, 4, 0)
self.layer43 = _DenseLayer(832, 32, 4, 0)
self.layer44 = _DenseLayer(864, 32, 4, 0)
self.layer45 = _DenseLayer(896, 32, 4, 0)
self.layer46 = _DenseLayer(928, 32, 4, 0)
self.layer47 = _DenseLayer(960, 32, 4, 0)
self.layer48 = _DenseLayer(992, 32, 4, 0)
self.layer49 = _Transition(num_input_features=1024, num_output_features=1024 // 2)
self.layer50 = _DenseLayer(512, 32, 4, 0)
self.layer51 = _DenseLayer(544, 32, 4, 0)
self.layer52 = _DenseLayer(576, 32, 4, 0)
self.layer53 = _DenseLayer(608, 32, 4, 0)
self.layer54 = _DenseLayer(640, 32, 4, 0)
self.layer55 = _DenseLayer(672, 32, 4, 0)
self.layer56 = _DenseLayer(704, 32, 4, 0)
self.layer57 = _DenseLayer(736, 32, 4, 0)
self.layer58 = _DenseLayer(768, 32, 4, 0)
self.layer59 = _DenseLayer(800, 32, 4, 0)
self.layer60 = _DenseLayer(832, 32, 4, 0)
self.layer61 = _DenseLayer(864, 32, 4, 0)
self.layer62 = _DenseLayer(896, 32, 4, 0)
self.layer63 = _DenseLayer(928, 32, 4, 0)
self.layer64 = _DenseLayer(960, 32, 4, 0)
self.layer65 = _DenseLayer(992, 32, 4, 0)
self.layer66 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer67 = nn.ReLU(inplace=True)
self.layer68 = nn.AvgPool2d(7, stride=1)
self.layer69 = nn.Linear(1024, 10, bias=True)'''

def get_caculations():
    return '''out1 = self.layer1(out0)
out2 = self.layer2(out1)
out3 = self.layer3(out2)
out4 = self.layer4(out3)
out5 = self.layer5(out4)
out6 = self.layer6(out5)
out7 = self.layer7(out6)
out8 = self.layer8(out7)
out9 = self.layer9(out8)
out10 = self.layer10(out9)
out11 = self.layer11(out10)
out12 = self.layer12(out11)
out13 = self.layer13(out12)
out14 = self.layer14(out13)
out15 = self.layer15(out14)
out16 = self.layer16(out15)
out17 = self.layer17(out16)
out18 = self.layer18(out17)
out19 = self.layer19(out18)
out20 = self.layer20(out19)
out21 = self.layer21(out20)
out22 = self.layer22(out21)
out23 = self.layer23(out22)
out24 = self.layer24(out23)
out25 = self.layer25(out24)
out26 = self.layer26(out25)
out27 = self.layer27(out26)
out28 = self.layer28(out27)
out29 = self.layer29(out28)
out30 = self.layer30(out29)
out31 = self.layer31(out30)
out32 = self.layer32(out31)
out33 = self.layer33(out32)
out34 = self.layer34(out33)
out35 = self.layer35(out34)
out36 = self.layer36(out35)
out37 = self.layer37(out36)
out38 = self.layer38(out37)
out39 = self.layer39(out38)
out40 = self.layer40(out39)
out41 = self.layer41(out40)
out42 = self.layer42(out41)
out43 = self.layer43(out42)
out44 = self.layer44(out43)
out45 = self.layer45(out44)
out46 = self.layer46(out45)
out47 = self.layer47(out46)
out48 = self.layer48(out47)
out49 = self.layer49(out48)
out50 = self.layer50(out49)
out51 = self.layer51(out50)
out52 = self.layer52(out51)
out53 = self.layer53(out52)
out54 = self.layer54(out53)
out55 = self.layer55(out54)
out56 = self.layer56(out55)
out57 = self.layer57(out56)
out58 = self.layer58(out57)
out59 = self.layer59(out58)
out60 = self.layer60(out59)
out61 = self.layer61(out60)
out62 = self.layer62(out61)
out63 = self.layer63(out62)
out64 = self.layer64(out63)
out65 = self.layer65(out64)
out66 = self.layer66(out65)
out67 = self.layer67(out66)
out68 = self.layer68(out67).view(1, -1)
out69 = self.layer69(out68)'''

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


   
# input_image = torch.randn(1, 3, 224, 224)

# densenet = DenseNet_()
# output_tensor = densenet(input_image)
# print(output_tensor.shape)