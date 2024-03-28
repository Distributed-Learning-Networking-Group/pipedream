# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
import torch
import re


def arch():
    return "resnet50"


def model(criterion):
    return [
        (Stage0(), ["input0"], ["out0", "out1"]),
        (Stage1(), ["out0", "out1"], ["out3", "out2"]),
        (Stage2(), ["out3", "out2"], ["out4"]),
        (criterion, ["out4"], ["loss"])
    ]


class Resnet50():
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


def model_resnet50(criterion, partition, recompute_ratio):
    _declares = get_declares()
    _calculations = get_caculations()
    module = Resnet50(_declares, _calculations)
    module.generate_layer_blocks()
    start = 0
    inputs = []
    outputs = [['out176']]
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
    return '''self.layer2 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
self.layer3 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer4 = torch.nn.ReLU()
self.layer5 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
self.layer6 = torch.nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer7 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer8 = torch.nn.ReLU()
self.layer9 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
self.layer10 = torch.nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer11 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer12 = torch.nn.ReLU()
self.layer13 = torch.nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer14 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer15 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer17 = torch.nn.ReLU()
self.layer18 = torch.nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer19 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer20 = torch.nn.ReLU()
self.layer21 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
self.layer22 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer23 = torch.nn.ReLU()
self.layer24 = torch.nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer25 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer27 = torch.nn.ReLU()
self.layer28 = torch.nn.Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer29 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer30 = torch.nn.ReLU()
self.layer31 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
self.layer32 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer33 = torch.nn.ReLU()
self.layer34 = torch.nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer35 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer37 = torch.nn.ReLU()
self.layer38 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
self.layer39 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer40 = torch.nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer41 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer42 = torch.nn.ReLU()
self.layer43 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
self.layer44 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer45 = torch.nn.ReLU()
self.layer46 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer47 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer49 = torch.nn.ReLU()
self.layer50 = torch.nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer51 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer52 = torch.nn.ReLU()
self.layer53 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
self.layer54 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer55 = torch.nn.ReLU()
self.layer56 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer57 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer59 = torch.nn.ReLU()
self.layer60 = torch.nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer61 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer62 = torch.nn.ReLU()
self.layer63 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
self.layer64 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer65 = torch.nn.ReLU()
self.layer66 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer67 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer68 = torch.nn.ReLU()
self.layer69 = torch.nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer70 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer71 = torch.nn.ReLU()
self.layer72 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
self.layer73 = torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer74 = torch.nn.ReLU()
self.layer75 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer76 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer78 = torch.nn.ReLU()
self.layer79 = torch.nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer80 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer81 = torch.nn.ReLU()
self.layer82 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
self.layer83 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer84 = torch.nn.ReLU()
self.layer85 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer86 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer87 = torch.nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
self.layer88 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer90 = torch.nn.ReLU()
self.layer91 = torch.nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer92 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer93 = torch.nn.ReLU()
self.layer94 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
self.layer95 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer96 = torch.nn.ReLU()
self.layer97 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer98 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer100 = torch.nn.ReLU()
self.layer101 = torch.nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer102 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer103 = torch.nn.ReLU()
self.layer104 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
self.layer105 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer106 = torch.nn.ReLU()
self.layer107 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer108 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer110 = torch.nn.ReLU()
self.layer111 = torch.nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer112 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer113 = torch.nn.ReLU()
self.layer114 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
self.layer115 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer116 = torch.nn.ReLU()
self.layer117 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer118 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer120 = torch.nn.ReLU()
self.layer121 = torch.nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer122 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer123 = torch.nn.ReLU()
self.layer124 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
self.layer125 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer126 = torch.nn.ReLU()
self.layer127 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer128 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer130 = torch.nn.ReLU()
self.layer131 = torch.nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer132 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer133 = torch.nn.ReLU()
self.layer134 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
self.layer135 = torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer136 = torch.nn.ReLU()
self.layer137 = torch.nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer138 = torch.nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer140 = torch.nn.ReLU()
self.layer141 = torch.nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer142 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer143 = torch.nn.ReLU()
self.layer144 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
self.layer145 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer146 = torch.nn.ReLU()
self.layer147 = torch.nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer148 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer149 = torch.nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
self.layer150 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer152 = torch.nn.ReLU()
self.layer153 = torch.nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer154 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer155 = torch.nn.ReLU()
self.layer156 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
self.layer157 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer158 = torch.nn.ReLU()
self.layer159 = torch.nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer160 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer162 = torch.nn.ReLU()
self.layer163 = torch.nn.Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer164 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer165 = torch.nn.ReLU()
self.layer166 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
self.layer167 = torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer168 = torch.nn.ReLU()
self.layer169 = torch.nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
self.layer170 = torch.nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
self.layer172 = torch.nn.ReLU()
self.layer173 = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
self.layer176 = torch.nn.Linear(in_features=2048, out_features=1000, bias=True)'''


def get_caculations():
    return '''out2 = self.layer2(out0)
out3 = self.layer3(out2)
out4 = self.layer4(out3)
out5 = self.layer5(out4)
out6 = self.layer6(out5)
out7 = self.layer7(out6)
out8 = self.layer8(out7)
out9 = self.layer9(out8)
out10 = self.layer10(out5)
out11 = self.layer11(out9)
out12 = self.layer12(out11)
out13 = self.layer13(out12)
out14 = self.layer14(out10)
out15 = self.layer15(out13)
out15 = out15 + out14
out17 = self.layer17(out15)
out18 = self.layer18(out17)
out19 = self.layer19(out18)
out20 = self.layer20(out19)
out21 = self.layer21(out20)
out22 = self.layer22(out21)
out23 = self.layer23(out22)
out24 = self.layer24(out23)
out25 = self.layer25(out24)
out25 = out25 + out17
out27 = self.layer27(out25)
out28 = self.layer28(out27)
out29 = self.layer29(out28)
out30 = self.layer30(out29)
out31 = self.layer31(out30)
out32 = self.layer32(out31)
out33 = self.layer33(out32)
out34 = self.layer34(out33)
out35 = self.layer35(out34)
out35 = out35 + out27
out37 = self.layer37(out35)
out38 = self.layer38(out37)
out39 = self.layer39(out38)
out40 = self.layer40(out37)
out41 = self.layer41(out40)
out42 = self.layer42(out41)
out43 = self.layer43(out42)
out44 = self.layer44(out43)
out45 = self.layer45(out44)
out46 = self.layer46(out45)
out47 = self.layer47(out46)
out39 = out39 + out47
out49 = self.layer49(out39)
out50 = self.layer50(out49)
out51 = self.layer51(out50)
out52 = self.layer52(out51)
out53 = self.layer53(out52)
out54 = self.layer54(out53)
out55 = self.layer55(out54)
out56 = self.layer56(out55)
out57 = self.layer57(out56)
out57 = out57 + out49
out59 = self.layer59(out57)
out60 = self.layer60(out59)
out61 = self.layer61(out60)
out62 = self.layer62(out61)
out63 = self.layer63(out62)
out64 = self.layer64(out63)
out65 = self.layer65(out64)
out66 = self.layer66(out65)
out67 = self.layer67(out66)
out67 = out67+out59
out68 = self.layer68(out67)
out69 = self.layer69(out68)
out70 = self.layer70(out69)
out71 = self.layer71(out70)
out72 = self.layer72(out71)
out73 = self.layer73(out72)
out74 = self.layer74(out73)
out75 = self.layer75(out74)
out76 = self.layer76(out75)
out76 = out76 + out68
out78 = self.layer78(out76)
out79 = self.layer79(out78)
out80 = self.layer80(out79)
out81 = self.layer81(out80)
out82 = self.layer82(out81)
out83 = self.layer83(out82)
out84 = self.layer84(out83)
out85 = self.layer85(out84)
out86 = self.layer86(out85)
out87 = self.layer87(out78)
out88 = self.layer88(out87)
out88 = out88 + out86
out90 = self.layer90(out88)
out91 = self.layer91(out90)
out92 = self.layer92(out91)
out93 = self.layer93(out92)
out94 = self.layer94(out93)
out95 = self.layer95(out94)
out96 = self.layer96(out95)
out97 = self.layer97(out96)
out98 = self.layer98(out97)
out98 = out98 + out90
out100 = self.layer100(out98)
out101 = self.layer101(out100)
out102 = self.layer102(out101)
out103 = self.layer103(out102)
out104 = self.layer104(out103)
out105 = self.layer105(out104)
out106 = self.layer106(out105)
out107 = self.layer107(out106)
out108 = self.layer108(out107)
out108 = out108 + out100
out110 = self.layer110(out108)
out111 = self.layer111(out110)
out112 = self.layer112(out111)
out113 = self.layer113(out112)
out114 = self.layer114(out113)
out115 = self.layer115(out114)
out116 = self.layer116(out115)
out117 = self.layer117(out116)
out118 = self.layer118(out117)
out118 = out118 + out110
out120 = self.layer120(out118)
out121 = self.layer121(out120)
out122 = self.layer122(out121)
out123 = self.layer123(out122)
out124 = self.layer124(out123)
out125 = self.layer125(out124)
out126 = self.layer126(out125)
out127 = self.layer127(out126)
out128 = self.layer128(out127)
out128 = out128 + out120
out130 = self.layer130(out128)
out131 = self.layer131(out130)
out132 = self.layer132(out131)
out133 = self.layer133(out132)
out134 = self.layer134(out133)
out135 = self.layer135(out134)
out136 = self.layer136(out135)
out137 = self.layer137(out136)
out138 = self.layer138(out137)
out138 = out138 + out130
out140 = self.layer140(out138)
out141 = self.layer141(out140)
out142 = self.layer142(out141)
out143 = self.layer143(out142)
out144 = self.layer144(out143)
out145 = self.layer145(out144)
out146 = self.layer146(out145)
out147 = self.layer147(out146)
out148 = self.layer148(out147)
out149 = self.layer149(out140)
out150 = self.layer150(out149)
out150 = out150 + out148
out152 = self.layer152(out150)
out153 = self.layer153(out152)
out154 = self.layer154(out153)
out155 = self.layer155(out154)
out156 = self.layer156(out155)
out157 = self.layer157(out156)
out158 = self.layer158(out157)
out159 = self.layer159(out158)
out160 = self.layer160(out159)
out160 = out160 + out152
out162 = self.layer162(out160)
out163 = self.layer163(out162)
out164 = self.layer164(out163)
out165 = self.layer165(out164)
out166 = self.layer166(out165)
out167 = self.layer167(out166)
out168 = self.layer168(out167)
out169 = self.layer169(out168)
out170 = self.layer170(out169)
out170 = out170 + out162
out172 = self.layer172(out170)
out173 = self.layer173(out172)
out174 = out173.size(0)
out175 = out173.view(out174, -1)
out176 = self.layer176(out175)'''
