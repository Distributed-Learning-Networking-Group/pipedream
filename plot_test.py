import numpy as np
import matplotlib.pyplot as plt

# 定义一个函数，将列表保存到文本文件中
def save_list_to_txt(data, filename):
    np.savetxt(filename, data)

# 定义一个函数，从文本文件中读取列表
def load_list_from_txt(filename):
    return np.loadtxt(filename)
def plot_multiple_curves(data_list):
    for data in data_list:
        plt.plot(data)
    plt.show()
# 定义一个函数，绘制曲线
def plot_curve(data):
    plt.plot(data)
    plt.show()


# 从文本文件中读取列表，并绘制曲线
loaded_list = load_list_from_txt("data_for_test.txt")
loaded_list1 = load_list_from_txt("data_for_test_1.txt")
plot_curve(loaded_list)
#plot_multiple_curves([loaded_list,loaded_list1])