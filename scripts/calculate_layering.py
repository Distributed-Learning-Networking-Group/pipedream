import math
import os
import numpy
import math
import model_detail
from torchvision.models import vgg16
import torch.nn as nn
# initial 动态规划目标矩阵layers stages replicate_time gpus
w = numpy.zeros((200, 10, 10, 10))
# transfer_channel = []  # perf 多机之间的带宽
# per layer sum time compute f+b 序号从1开始
# parameter_size = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 每层的梯度大小
process_time = []  # 前向计算与方向计算时间之和
DNN_per_layer_compute_time_foward = []  # 每层的前向计算时间
DNN_per_layer_compute_time_backward = []  # 每层的反向计算时间
DNN_per_layer_activation = []  # 每层的激活量 前向给后向传的东西 out_put_size

DNN_per_layer_gradients = []  # 每层更新时候的梯度 optimizer.step()呢部分的

GPUS_order_list = [0, 0, 1, 2, 3]  # gpu之间带宽由高到低，里面应该是gpu的id
GPUS_bandwidth = []  # 10Gbps to MByte


PIPEDREAM_FLAG = 1
GPU_NUM = 4
DIR = "profile"
DEP_IDX = 2


def read_file(filename):
    list = []
    with open(DIR+'/'+filename) as file:
        for line in file:
            list.append(line)
    return list


def data_init(layers_detail):
    file_names = os.listdir(DIR)

    # init DNN_per_layer_compute_time
    for file_index in range(0, len(file_names)):
        if 'bac_.txt' in file_names[file_index]:
            list = read_file(file_names[file_index])
            for i in list:
                DNN_per_layer_compute_time_backward.append(float(i))
        elif 'for_.txt' in file_names[file_index]:
            list = read_file(file_names[file_index])
            for i in list:
                DNN_per_layer_compute_time_foward.append(float(i))

    # init process_time
    for index in range(0, len(DNN_per_layer_compute_time_foward)):
        process_time.append(
            DNN_per_layer_compute_time_foward[index]+DNN_per_layer_compute_time_backward[index])

    # init DNN_per_layer_activation
    for index in range(0, len(layers_detail[DEP_IDX]['output_shape'])):
        per_layer_shape = layers_detail[DEP_IDX]['output_shape'][index]
        per_layer_size = math.prod(per_layer_shape) * \
            32/8/1024/1024  # shape*float32 to MByte
        DNN_per_layer_activation.append(per_layer_size)

    # init DNN_per_layer_gradients
    for index in range(0, len(layers_detail[DEP_IDX]['kernel_shape'])):
        per_layer_gradients = layers_detail[DEP_IDX]['kernel_shape'][index]
        per_layer_gradients_size = math.prod(per_layer_gradients) * \
            32/8/1024/1024  # shape*float32 to MByte
        DNN_per_layer_gradients.append(per_layer_gradients_size)

    # init GPUS_bandwidth
    for index in range(0, GPU_NUM):
        GPUS_bandwidth.append([1280, 1280, 1280, 1280])


def compute_all_reduce_time(ngpus, start_layer, end_layer, set_gpu):
    if ngpus == 0:
        return 0
    sum_parameter = 0
    # 看ddp交换的是什么，梯度还是参数
    for i in range(start_layer, end_layer+1):
        sum_parameter += DNN_per_layer_gradients[i]
    mini_bandwidth = float('inf')
    for index1, i in enumerate(set_gpu):
        for index2, j in enumerate(set_gpu):
            if GPUS_bandwidth[i][j] < mini_bandwidth:
                mini_bandwidth = GPUS_bandwidth[i][j]
    time = 2*(ngpus-1)*(sum_parameter)/(ngpus*mini_bandwidth)
    return time


def compute_communication_time(gpu_stage1, gpu_stage2, layer1_id):
    mini_bandwidth = float('inf')
    for index1, i in enumerate(gpu_stage1):
        for index2, j in enumerate(gpu_stage2):
            if GPUS_bandwidth[i][j] < mini_bandwidth:
                mini_bandwidth = GPUS_bandwidth[i][j]
    time = ((DNN_per_layer_activation[layer1_id]) /
            (len(gpu_stage1)*len(gpu_stage2)*mini_bandwidth))
    return time


def compute_stage_partition(layers, n_gpus, stages, replicate_times, F_match={}, Sets=[]):
    if layers < stages or n_gpus < stages:
        return float('inf'), [], {}
    if stages == 1 and replicate_times == n_gpus:
        time_ = 0
        for i in range(1, layers+1):
            time_ += process_time[i]*(1/n_gpus)
        time_ += compute_all_reduce_time(
            n_gpus, 1, layers, GPUS_order_list[1:n_gpus+1]
        )
        w[layers][stages][replicate_times][n_gpus] = time_
        Sets = list(range(1, layers+1))
        F_match[tuple(Sets)] = tuple(GPUS_order_list[1:n_gpus+1])
        # return w[layers][stages][replicate_times][n_gpus],Sets,F_match
    else:
        if stages == 1 or replicate_times == n_gpus:
            return float('inf'), [], {}
    min_max_time = float('inf')
    if stages == 1 and replicate_times == n_gpus:
        min_max_time = w[layers][stages][replicate_times][n_gpus]

    F_match_ = {}
    for L in range(1, layers):
        for R in range(1, n_gpus-replicate_times+1):
            w[L][stages-1][R][n_gpus-replicate_times], Sets_, F_match = compute_stage_partition(
                L, n_gpus-replicate_times, stages-1, R)
            communication_time = compute_communication_time(
                GPUS_order_list[n_gpus-replicate_times-R+1:n_gpus-replicate_times +
                                1], GPUS_order_list[n_gpus-replicate_times+1:n_gpus+1], L
            )
            process_times = 0
            for i in range(L+1, layers+1):
                process_times += process_time[i]*(1/replicate_times)
            rest_time = process_times+compute_all_reduce_time(
                replicate_times, L+1, layers, GPUS_order_list[n_gpus-replicate_times+1:n_gpus+1])
            if communication_time < (1/2)*abs((rest_time-w[L][stages - 1][R][n_gpus - replicate_times])):
                max_time = max(
                    rest_time, w[L][stages - 1][R][n_gpus - replicate_times])
            else:
                if rest_time > w[L][stages-1][R][n_gpus-replicate_times]:
                    max_time = max(rest_time, communication_time+rest_time-(1/2)
                                   * (rest_time-w[L][stages - 1][R][n_gpus - replicate_times]))
                else:
                    max_time = max(w[L][stages - 1][R][n_gpus - replicate_times], communication_time + w[L][stages - 1][R][n_gpus - replicate_times] - (1 / 2) * (
                        w[L][stages - 1][R][n_gpus - replicate_times] - rest_time))
            if max_time < min_max_time:
                min_max_time = max_time
                s = list(range(L+1, layers+1))
                Sets = [Sets_, s]
                F_match[tuple(s)] = tuple(
                    GPUS_order_list[n_gpus-replicate_times+1:n_gpus+1])
                F_match_ = dict(F_match)
    return min_max_time, Sets, F_match_


def compute_stage_partition_pipedream(layers, n_gpus, stages, replicate_times, F_match={}, Sets=[]):
    if layers < stages or n_gpus < stages:
        return float('inf'), [], {}
    if stages == 1 and replicate_times == n_gpus:
        time_ = 0
        for i in range(1, layers+1):
            time_ += process_time[i]*(1/n_gpus)
        time_ += compute_all_reduce_time(
            n_gpus, 1, layers, GPUS_order_list[1:n_gpus+1]
        )
        w[layers][stages][replicate_times][n_gpus] = time_
        Sets = list(range(1, layers+1))
        F_match[tuple(Sets)] = tuple(GPUS_order_list[1:n_gpus+1])
        # return w[layers][stages][replicate_times][n_gpus],Sets,F_match
    else:
        if stages == 1 or replicate_times == n_gpus:
            return float('inf'), [], {}
    min_max_time = float('inf')
    if stages == 1 and replicate_times == n_gpus:
        min_max_time = w[layers][stages][replicate_times][n_gpus]

    F_match_ = {}
    for L in range(1, layers):
        for R in range(1, n_gpus-replicate_times+1):
            w[L][stages-1][R][n_gpus-replicate_times], Sets_, F_match = compute_stage_partition(
                L, n_gpus-replicate_times, stages-1, R)
            communication_time = compute_communication_time(
                GPUS_order_list[n_gpus-replicate_times-R+1:n_gpus-replicate_times +
                                1], GPUS_order_list[n_gpus-replicate_times+1:n_gpus+1], L
            )
            process_times = 0
            for i in range(L+1, layers+1):
                process_times += process_time[i]*(1/replicate_times)
            rest_time = process_times+compute_all_reduce_time(
                replicate_times, L+1, layers, GPUS_order_list[n_gpus-replicate_times+1:n_gpus+1])
            max_time = max(rest_time, 2*communication_time,
                           w[L][stages - 1][R][n_gpus - replicate_times])

            if max_time < min_max_time:
                min_max_time = max_time
                s = list(range(L+1, layers+1))
                Sets = [Sets_, s]
                F_match[tuple(s)] = tuple(
                    GPUS_order_list[n_gpus-replicate_times+1:n_gpus+1])
                F_match_ = dict(F_match)
    return min_max_time, Sets, F_match_


if __name__ == '__main__':
    # w(layers, sum_gpus, stage_num, replicate_times)
    layers_detail = model_detail.model_info(
        vgg16(pretrained=False), batch_size=16).layers_info
    data_init(layers_detail)

    for i in range(1, 1+GPU_NUM):
        for j in range(1, 1+GPU_NUM):
            time, sets, match = compute_stage_partition(10, 4, i, j, {}, [])
            # print(i, j)
            print(f'time:{time}')
            print(f'sets:{sets}')
            print(f'match:{match}\n')
