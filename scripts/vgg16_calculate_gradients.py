import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16
import model_detail

# 1. 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# 2. 定义模型
net = vgg16(pretrained=False)  # 如果要用预训练模型，将False改为True
net.classifier[6] = torch.nn.Linear(4096, 10)  # 修改分类器以适应CIFAR10

# 3. 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4. 训练模型
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # 每2000个batch打印一次训练状态
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

        # 打印梯度
        for name, parameter in net.named_parameters():
            if parameter.grad is not None:
                print(f'Layer: {name} | Gradient: {parameter.grad}')

print('Finished Training')
