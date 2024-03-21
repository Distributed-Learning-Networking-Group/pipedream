import torch
import torchvision.models as models
import time
# 加载预训练的ResNet-50模型
if torch.cuda.is_available():
    device = torch.device("cuda:0")
model = models.resnet50().to(device)
model.eval()  # 设置模型为评估模式

# 生成随机输入数据
input_tensor = torch.rand(64, 3, 224, 224).to(device)  # 生成一个形状为[1, 3, 224, 224]的随机张量，代表一个RGB图像
# torch.cuda.synchronize()
# 进行1000次前向推理
start_time = time.time()
for i in range(30000000):
    with torch.no_grad():  # 禁用梯度计算
        output = model(input_tensor)
# torch.cuda.synchronize()
end_time = time.time()

# 计算平均推理时间
average_inference_time = (end_time - start_time)
print('inference time:', average_inference_time, 'seconds')
# import os
#
# import torch
# import torch.distributed as dist
# import argparse
# def send_dict(dict_obj, dst, tag):
#     # 序列化字典为字节
#     buffer = torch.ByteTensor(bytearray(str(dict_obj), 'utf-8'))
#     # 发送字节长度信息
#     size_tensor = torch.tensor([buffer.numel()], dtype=torch.long)
#     dist.send(tensor=size_tensor, dst=dst, tag=tag)
#     # 发送字节数据
#     dist.send(tensor=buffer, dst=dst, tag=tag+1)
#
# def recv_dict(src, tag):
#     # 接收字节长度信息
#     size_tensor = torch.tensor([0], dtype=torch.long)
#     dist.recv(tensor=size_tensor, src=src, tag=tag)
#     # 接收字节数据
#     buffer = torch.ByteTensor(size_tensor.item())
#     dist.recv(tensor=buffer, src=src, tag=tag+1)
#     # 反序列化字节到字典
#     dict_obj = eval(buffer.numpy().tobytes().decode('utf-8'))
#     return dict_obj
#
# # 示例：进程0发送字典给进程1
# parser = argparse.ArgumentParser(
#         description='Test lightweight communication library')
# parser.add_argument("--master_addr", required=True, type=str,
#                         help="IP address of master")
# parser.add_argument("--rank", required=True, type=int,
#                         help="Rank of current worker")
# parser.add_argument('-p', "--master_port", default=12345,
#                         help="Port used to communicate tensors")
#
# args = parser.parse_args()
# os.environ['MASTER_ADDR'] = args.master_addr
# os.environ['MASTER_PORT'] = args.master_port
# dist.init_process_group(backend="gloo",rank=args.rank,world_size=2)
# rank = dist.get_rank()
# if rank == 0:
#     # 创建一个字典
#     my_dict = {'a': 1, 'b': 2}
#     # 发送字典到进程1
#     send_dict(my_dict, dst=1, tag=0)
# elif rank == 1:
#     # 从进程0接收字典
#     received_dict = recv_dict(src=0, tag=0)
#     print('Received dictionary:', received_dict)
