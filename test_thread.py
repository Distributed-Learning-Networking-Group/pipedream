import torch
import numpy
# 创建一个张量
tensor = torch.randn (32, 128, 56, 56)

# 计算张量中元素的总数
num_elements = tensor.numel()

# 获取张量元素的数据类型的字节大小
element_size = tensor.element_size()

# 计算张量的总内存占用（以字节为单位）
memory_bytes = num_elements * element_size

print(f"张量的内存占用为：{memory_bytes/1024/1024} 字节")

