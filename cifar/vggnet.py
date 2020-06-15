import torch.nn as nn
import torch.nn.functional as F



class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()

        # 3 * 28 * 28
        self.con1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.max_pooling1 = nn.MaxPool2d(kernel_size=2, stride= 2)

        # 3 * 14 * 14
        self.con2_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.con2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.max_pooling2 = nn.MaxPool2d(kernel_size=2, stride= 2)

        # 3 * 7 * 7
        self.con3_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.con3_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.max_pooling3 = nn.MaxPool2d(kernel_size=2, stride=2,
                                         padding= 1)  # 7 * 7  --> 3 会丢失边缘信息，padding + 1, 8 * 8 --> 4

        # 3 * 4 * 4
        self.con4_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.con4_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.max_pooling4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # batch_size * 512 * 2 * 2
        self.fc = nn.Linear(512 * 4, 10)

    def forword(self, x):

        batchsize = x.size(0)
        out = self.con1(x)
        out = self.max_pooling1(out)

        out = self.con2_1(out)
        out = self.con2_2(out)
        out = self.max_pooling2(out)

        out = self.con3_1(out)
        out = self.con3_2(out)
        out = self.max_pooling3(out)

        out = self.con4_1(out)
        out = self.con4_2(out)
        out = self.max_pooling4(out)

        out = out.view(batchsize, -1)

        # batchsze * c * h * w --> batchsize * n
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
        return out


def VggBase():
    return VGGNet()
