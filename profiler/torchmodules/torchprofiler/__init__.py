# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .profiling import Profiling

import torch
import torch.distributed as dist

# 假设已经初始化了进程组
# dist.init_process_group(...)

def send_dict(dict_obj, dst, tag):
    # 序列化字典为字节
    buffer = torch.ByteTensor(bytearray(str(dict_obj), 'utf-8'))
    # 发送字节长度信息
    size_tensor = torch.tensor([buffer.numel()], dtype=torch.long)
    dist.send(tensor=size_tensor, dst=dst, tag=tag)
    # 发送字节数据
    dist.send(tensor=buffer, dst=dst, tag=tag+1)

def recv_dict(src, tag):
    # 接收字节长度信息
    size_tensor = torch.tensor([0], dtype=torch.long)
    dist.recv(tensor=size_tensor, src=src, tag=tag)
    # 接收字节数据
    buffer = torch.ByteTensor(size_tensor.item())
    dist.recv(tensor=buffer, src=src, tag=tag+1)
    # 反序列化字节到字典
    dict_obj = eval(buffer.numpy().tobytes().decode('utf-8'))
    return dict_obj

# 示例：进程0发送字典给进程1
rank = dist.get_rank()
if rank == 0:
    # 创建一个字典
    my_dict = {'a': 1, 'b': 2}
    # 发送字典到进程1
    send_dict(my_dict, dst=1, tag=0)
elif rank == 1:
    # 从进程0接收字典
    received_dict = recv_dict(src=0, tag=0)
    print('Received dictionary:', received_dict)
