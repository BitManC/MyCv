from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import torch
import glob

label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
label_dict = {}
for idx, name in enumerate(label_name):
    label_dict[name] = idx


def default_loader(path):
    return Image.open(path).convert("RGB")

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop((28, 28)),  # resize
    # transforms.RandomHorizontalFlip(),  # 翻转
    # transforms.RandomVerticalFlip(),  # 翻转
    transforms.RandomRotation(90),
    transforms.RandomGrayscale(0.1),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

class MyDataset(Dataset):

    def __init__(self, im_list, transform=None, loader=default_loader):
        '''
        :param im_list: data list
        :param transform: 数据增强
        :param loader:
        '''
        super(MyDataset, self).__init__()
        imgs = []
        for im_item in im_list:
            im_label_name = im_item.split('/')[-2]
            imgs.append([im_item, label_dict[im_label_name]])

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        # 数据增强部分
        im_path, im_label = self.imgs[index]
        im_data = self.loader(im_path)

        if self.transform is not None:
            im_data = self.transform(im_data)
        return im_label, im_data

    def __len__(self):
        return len(self.imgs)


im_train_list = glob.glob(r'/Users/yifz/CV/MyCv/cifar/cifar_set/train/*/*.png')
im_test_list = glob.glob(r"/Users/yifz/CV/MyCv/cifar/cifar_set/train/*/*.png")

train_dataset = MyDataset(im_train_list, transform=train_transforms)
test_dataset = MyDataset(im_test_list, transform=test_transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=6, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=6, shuffle=True, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# for i, data in enumerate(train_loader):
#     labels, inputs = data
#     inputs, labels = inputs.to(device), labels.to(device)
#     print(type(inputs))