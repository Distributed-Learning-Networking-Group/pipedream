import argparse
import torch
import torch.optim as optim
from torchvision import transforms
from torchvision import models
from torchvision import datasets
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 2)')
args = parser.parse_args()
batch_size=args.batch_size
if torch.cuda.is_available():
    device = torch.device("cuda:0")
model=models.vgg16().to(device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)


# 4. 设置训练参数和优化器
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# model.eval()
# with torch.no_grad():
while True:
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
