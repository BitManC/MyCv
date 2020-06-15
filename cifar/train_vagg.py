import torch
import torch.nn as nn
import torchvision
from vggnet import VGGNet
from load_cifar import train_loader, test_loader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch_num = 200
lr = 0.01

net = VGGNet().to(device)

# loss
loss_func = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)  # 自动调节lr

for epoch in range(epoch_num):
    print("epoch is :", epoch)
    net.train()

    for i, data in enumerate(train_loader):
        print("step", i)
        labels, inputs = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)   # error
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("step ", i, "'s loss is:", loss.item())
