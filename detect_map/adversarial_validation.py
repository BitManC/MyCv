from utils import parse_images
from utils import ImgMinMaxScaler
from utils import ImgResize
from utils import ImgToTensor
from utils import AdversarialDataSet
import pandas as pd
from pathlib import Path
import torchvision
import torch.nn as nn
import torch
import tqdm
import numpy as np
from torch.utils.data import DataLoader
import os

'''
查看测试集训练集的分布情况
'''



rootDir = Path(r"D:/Users/yifz/kaggle/unet/dstl")
path_merge = lambda x: os.path.join(rootDir.x)


DF = pd.read_csv(path_merge('train_wkt_v4.csv.zip'))
GS = pd.read_csv(path_merge('grid_sizes.csv.zip'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

threeBandImagesDir = path_merge("three_band")
sixteenBandImagesDir = path_merge("sixteen_band")
df_train_val_all_classes = pd.read_csv(path_merge("train_wkt_v4.csv.zip"))

imagenet_means = [0.485, 0.456, 0.406]
imagenet_stds = [0.229, 0.224, 0.225]

df_img_data = parse_images(threeBandImagesDir)
df_train_val_img_data = df_img_data.loc[df_img_data.ImageId.isin(df_train_val_all_classes.ImageId.unique())]
df_test_img_data = df_img_data.loc[~df_img_data.ImageId.isin(df_train_val_all_classes.ImageId.unique())]

img_mins = np.min(df_img_data.img_channel_min.to_list(), axis=0)
img_maxs = np.max(df_img_data.img_channel_max.to_list(), axis=0)

tf_adversarial = torchvision.transforms.Compose([
    ImgMinMaxScaler(channel_min=img_mins, channel_max=img_maxs),
    ImgResize((300, 300)),
    ImgToTensor(),
    torchvision.transforms.Normalize(mean=imagenet_means, std=imagenet_stds)
])

ds_adversarial = AdversarialDataSet(train_img_ids=df_train_val_img_data.ImageId,
                                    test_img_ids=df_test_img_data.ImageId,
                                    transform=tf_adversarial)

dl_adversarial = DataLoader(ds_adversarial, batch_size=32, num_workers=0)

adv_model = torchvision.models.resnet18(pretrained=True)
adv_model.fc = nn.Sequential(nn.Linear(512, 1))
adv_model.to("cuda")
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(adv_model.parameters(), lr=1e-3)


def train(num_epochs, model=adv_model, criterion=criterion, optimizer=optimizer, dl=dl_adversarial, device="cuda"):
    model.train()
    for epoch in range(num_epochs, desc="Epochs", leave=False):
        running_loss = 0
        for img, label in tqdm(dl, desc=f"Epoch {epoch}", leave=False, position=0):
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()

            pred_logits = model(img).squeeze()

            loss = criterion(pred_logits, label.type(torch.float32))

            running_loss += loss.item() * img.size(0) / len(dl.dataset)

            loss.backward()
            optimizer.step()

        tqdm.write(f"Epoch {epoch}: loss={running_loss:.6f}")

if __name__ == '__main__':
    train(6)
