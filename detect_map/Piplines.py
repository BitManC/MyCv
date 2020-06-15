import torchvision
from pathlib import Path
import pandas as pd
import numpy as np
from utils import TiffImgDataset
from utils import parse_images
from utils import ImgResize
from utils import ImgToTensor
from utils import parse_mask_data
from utils import stack
from utils import calculate_channel_means_and_stds
from utils import Cropper
from utils import ImgMaskTransformCompose
from utils import ImgMaskMinMaxScaler
from utils import ImgMinMaxScaler
from utils import ImgMaskTensorNormalize
from utils import ImgMaskRandomHorizontalFlip
from utils import ImgMaskRandomVerticalFlip
from utils import ImgMaskToTensor
from torch.utils.data import DataLoader
import tqdm
from DataLoader import SatelliteDataset
from Unet import UNet
from Trainner import Trainer
from Trainner import AccuracyMetric
from Trainner import PrecisionRecallF1Metric
from Trainner import IOUMetric
from Trainner import DiceBCELoss
import torch


rootDir = Path("/Users/yifz/kaggle/unet/dstl")
DF = pd.read_csv(rootDir / 'train_wkt_v4.csv.zip')
GS = pd.read_csv(rootDir / 'grid_sizes.csv.zip', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
threeBandImagesDir = rootDir / "three_band"
sixteenBandImagesDir = rootDir / "sixteen_band"
df_train_val_all_classes = pd.read_csv(rootDir / "train_wkt_v4.csv.zip")

imagenet_means = [0.485, 0.456, 0.406]
imagenet_stds = [0.229, 0.224, 0.225]

df_img_data = parse_images(threeBandImagesDir)

df_train_val_img_data = df_img_data.loc[df_img_data.ImageId.isin(df_train_val_all_classes.ImageId.unique())]
df_train_val_buildings = df_train_val_all_classes[df_train_val_all_classes.ClassType == 1]
df_train_val_buildings = parse_mask_data(df_train_val_buildings)
df_train_val_buildings = df_train_val_buildings.join(df_train_val_img_data.set_index('ImageId'), on='ImageId')
df_train_val_buildings.to_csv("df_train_val_buildings.csv", index=False)

val_img_ids = ["6100_2_2", "6140_3_1"]
df_val_buildings = df_train_val_buildings[df_train_val_buildings.ImageId.isin(val_img_ids)]
df_train_buildings = df_train_val_buildings[~df_train_val_buildings.ImageId.isin(val_img_ids) & df_train_val_buildings.NumPolygons != 0]

trainValPreprocessedDir = rootDir / "preprocessed"
croppedImgDir = rootDir / "cropped_imgs"
croppedMaskDir = rootDir / "cropped_masks"
trainCroppedImgDir = croppedImgDir / "training_set"   # train input
trainCroppedMaskDir = croppedMaskDir / "training_set"  # train label
validationCroppedImgDir = croppedImgDir / "validation_set"
validationCroppedMaskDir = croppedMaskDir / "validation_set"

for directory in [croppedImgDir, croppedMaskDir, trainCroppedImgDir,
                  trainCroppedMaskDir, validationCroppedImgDir,
                  validationCroppedMaskDir]:
    if not directory.exists():
        directory.mkdir()

training_cropper = Cropper(crop_height=300, crop_width=300, n_crops=50)
training_cropper.crop_all(df_train_buildings.ImageId, trainValPreprocessedDir,
                          trainCroppedImgDir, trainCroppedMaskDir)

# Define validation cropper, here we do not use random crops.
validation_cropper = Cropper(crop_height=1110, crop_width=1110)

# Create validation crops
validation_cropper.crop_all(df_val_buildings.ImageId, trainValPreprocessedDir, validationCroppedImgDir, validationCroppedMaskDir)

# Trainings and validation transformations
tf_train = ImgMaskTransformCompose([
     ImgMaskMinMaxScaler(img_mins=0, img_maxs=2047),
     ImgMaskRandomHorizontalFlip(),
     ImgMaskRandomVerticalFlip(),
     ImgMaskToTensor(),
     ImgMaskTensorNormalize(mean=imagenet_means, std=imagenet_stds)
     ]
)

tf_valid = ImgMaskTransformCompose([
    ImgMaskMinMaxScaler(img_mins=0, img_maxs=2047),
    ImgMaskToTensor(),
    ImgMaskTensorNormalize(mean=imagenet_means, std=imagenet_stds)
    ])

class CropRecreator:
    def __init__(self, cropper=training_cropper, image_ids=df_train_buildings.ImageId,
                 src_img_dir=trainValPreprocessedDir,
                 crop_image_dir=trainCroppedImgDir, crop_mask_dir=trainCroppedMaskDir,
                 recreation_epoch_frequency=2):
        self.cropper = training_cropper
        self.image_ids = image_ids
        self.crop_image_dir = crop_image_dir
        self.crop_mask_dir = crop_mask_dir
        self.recreation_epoch_frequency = recreation_epoch_frequency
        self.src_image_dir = src_img_dir

    def __call__(self, epoch):
        if (epoch + 1) % self.recreation_epoch_frequency == 0:
            self.cropper.crop_all(self.image_ids, self.src_image_dir, self.crop_image_dir, self.crop_mask_dir)
            tqdm.write("Recreated random crops!")

# Training data

ds_train = SatelliteDataset(trainCroppedImgDir, trainCroppedMaskDir, transform=tf_train)
ds_valid = SatelliteDataset(validationCroppedImgDir, validationCroppedMaskDir, transform=tf_valid)

# DataLoader
BATCH_SIZE = 24
modelSaveDir = rootDir.parent / "models"

dl_train = DataLoader(ds_train,
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      num_workers=0)

dl_valid = DataLoader(ds_valid,
                      batch_size=8,
                      num_workers=0)

unet34 = UNet(n_classes=1,
            input_channels=3,
            adaptiveInputPadding=True,
            pretrained_encoder=True,
            freeze_encoder=False,
            use_hypercolumns=True,
            use_attention=False,
            use_recurrent_decoder_blocks=False,
            dropout=0.2)

training_building_area_ratio = df_train_buildings.BuildingAreaRatio.mean()

building_class_weight = (1.0 - training_building_area_ratio) / training_building_area_ratio

trainer = Trainer(unet34,
                  dl_train,
                  DiceBCELoss(bce_pos_weight=building_class_weight),
                  torch.optim.Adam(params=unet34.parameters(), lr=1e-7, weight_decay=1e-4),
                  dl_valid=dl_valid,
                  metrics=[IOUMetric(), AccuracyMetric(), PrecisionRecallF1Metric()],
                  on_train_val_epoch_finished_callback=CropRecreator())
