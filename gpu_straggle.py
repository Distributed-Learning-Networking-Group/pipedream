import time
import numpy
import torch

def save_list_to_txt(data, filename):
    numpy.savetxt(filename, data)
def main():
    # 检查GPU是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # 创建一个大张量，用于占用GPU
    big_tensor = torch.randn(5000, 5000).to(device)

    # 进行一些计算，以增加GPU占用率

    for i in range(10000):
        print(i)
        result = torch.matmul(big_tensor, big_tensor)

    del big_tensor
if __name__ == '__main__':
    main()
