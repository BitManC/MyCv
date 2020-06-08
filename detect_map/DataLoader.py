import os
import tifffile
from PIL import Image
import matplotlib.pyplot as  plt
from torch.utils.data import Dataset, DataLoader
from utils import stretch_n


class SatelliteDataset(Dataset):
    '''
    Create statellite dataset, implement torch.utils.data.Dataset
    '''

    def __init__(self, imgDir, maskDir, transform=None):
        super(SatelliteDataset, self).__init__()
        self.imgDir = imgDir
        self.maskDir = maskDir
        self.transform = transform
        self.imageFiles = sorted([item for item in imgDir.iterdir() if item.is_file()])

    def __getitem__(self, index):
        img_path = self.imageFiles[index]
        mask_path = self.maskDir / f"{img_path.stem}.png"
        img = tifffile.imread(os.path.join(img_path))
        mask = Image.open(mask_path)
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        return img, mask
    # def show_untransformed_sample(self, index, maskAlpha=0.3, figsize=(8, 8), rgb_channels=[0, 1, 2]):
    #     img_path = self.imageFiles[index]
    #     mask_path = self.maskDir / f"{img_path.stem}.png"
    #
    #     img = stretch_n(tifffile.imread(os.path.join(img_path))[:, :, rgb_channels])
    #     mask = Image.open(mask_path)
    #
    #     fig, ax = plt.subplots(figsize=figsize)
    #     tifffile.imshow(img, figure=fig, subplot=ax)
    #     ax.imshow(mask, alpha=maskAlpha)
    #     ax.axis("off")
    #     plt.show()

    def __len__(self):
        return len(self.imageFiles)