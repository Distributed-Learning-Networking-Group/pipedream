import torchvision.models
from torchsummary import summary  # pip install torch-summary
import torch.nn as nn
import re
test_model = torchvision.models.resnet50(pretrained=False)


class model_info():
    """
    Atributtes:
        model:需要分析的模型
        batch_size:训练时选用的batch_sizes

    model_info.layers_info[summary中的depth-idx][具体参数]
    """

    def __init__(self, model: nn.modules, batch_size: int):
        self.batch_size = batch_size
        self.info = summary(model, (3, 224, 224), col_names=[
                            "input_size", "output_size", "num_params", "kernel_size", "mult_adds"])
        self.layers_info = []
        for _ in range(0, 4):
            self.layers_info.append({'layer_name': [],
                                    'input_shape': [],
                                     'output_shape': [],
                                     'param': [],
                                     'kernel_shape': [],
                                     'mult_adds': [],
                                     'total_params': int,
                                     'trainable_params': int,
                                     'non_trainable_params': int,
                                     'total_mult_adds(G)': float,
                                     'input_size(MB)': float,
                                     'for/bac_pass_size(MB)': float,
                                     'params_size(MB)': float,
                                     'estimated_total_size(MB)': float})
        self._info_split()

    def _str_list_to_list(self, str_list):
        if str_list == '--':
            return [0]
        list = str_list[1:-1].split(', ')
        list = [int(i) if int(i) != -1 else self.batch_size for i in list]
        return list

    def _info_split(self):
        layers = str(self.info).split('\n')
        layers[0] = ''
        layers[1] = ''
        layers[2] = ''
        layers[-1] = ''
        layers[-6] = ''
        layers[-11] = ''

        for index in range(3, len(layers)-11):
            # 根据depth-idx区分layer类型
            dep_idx = int(re.search(r':\s*([^\s])', layers[index]).group(1))

            layers[index] = re.split(r' {2,}', layers[index].split('─')[1])

            self.layers_info[dep_idx]['layer_name'].append(layers[index][0])
            self.layers_info[dep_idx]['input_shape'].append(
                self._str_list_to_list(layers[index][1]))
            self.layers_info[dep_idx]['output_shape'].append(
                self._str_list_to_list(layers[index][2]))
            if layers[index][3] == '--':
                self.layers_info[dep_idx]['param'].append(0)
            else:
                self.layers_info[dep_idx]['param'].append(
                    int(layers[index][3].replace(',', "")))
            self.layers_info[dep_idx]['kernel_shape'].append(
                self._str_list_to_list(layers[index][4]))
            if layers[index][5] == '--':
                self.layers_info[dep_idx]['mult_adds'].append(0)
            else:
                self.layers_info[dep_idx]['mult_adds'].append(
                    int(layers[index][5].replace(',', "")))

        self.layers_info[dep_idx]['total_params'] = int(
            layers[-10].split()[-1].replace(',', ""))
        self.layers_info[dep_idx]['trainable_params'] = int(
            layers[-9].split()[-1].replace(',', ""))
        self.layers_info[dep_idx]['non_trainable_params'] = int(
            layers[-8].split()[-1].replace(',', ""))
        self.layers_info[dep_idx]['total_mult_adds(G)'] = float(
            layers[-7].split()[-1])

        self.layers_info[dep_idx]['input_size(MB)'] = float(
            layers[-5].split()[-1])
        self.layers_info[dep_idx]['for/bac_pass_size(MB)'] = float(
            layers[-4].split()[-1])
        self.layers_info[dep_idx]['params_size(MB)'] = float(
            layers[-3].split()[-1])
        self.layers_info[dep_idx]['estimated_total_size(MB)'] = float(
            layers[-2].split()[-1])


# if __name__ == '__main__':
#     layers_info = model_info(test_model, batch_size=16)
