import numpy as np
from shapely.wkt import loads as wkt_loads
from pathlib import Path
import cv2
import tqdm
import os
import tifffile
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
import torchvision
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
import random
from geopandas import GeoSeries
from shapely.geometry import Polygon
from PIL import Image


rootDir = Path("/Users/yifz/kaggle/unet/dstl")
DF = pd.read_csv(rootDir / 'train_wkt_v4.csv.zip')
GS = pd.read_csv(rootDir / 'grid_sizes.csv.zip', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)


def stack(band_paths, out_path):
    """Stack a set of bands into a single file.
    Based on: https://github.com/mapbox/rasterio/issues/1273#issue-295230001

    Parameters
    ----------
    band_paths : list of file paths
        A list with paths to the bands you wish to stack. Bands
        will be stacked in the order given in this list.
    out_path : string
        A path for the output file.
    """
    first_band = rasterio.open(band_paths[0], 'r')
    meta = first_band.meta.copy()

    counts = 0
    for ifile in band_paths:
        first_band = rasterio.open(band_paths[0], 'r')
        with rasterio.open(ifile, 'r') as ff:
            counts += ff.meta['count']
    meta.update(count=counts)

    band_start_index = 1

    with rasterio.open(out_path, 'w', **meta) as ff:
        for img_path in band_paths:
            with rasterio.open(img_path, 'r') as band_set:
                read_band_idx = range(1, 1 + band_set.count)
                bands = band_set.read(read_band_idx,
                                      out_shape=(band_set.count, meta["height"], meta["width"]),
                                      resampling=Resampling.bilinear)
                if bands.ndim != 3:
                    bands = bands[np.newaxis, ...]
                write_band_idx = range(band_start_index, band_start_index + band_set.count)
                ff.write(bands, write_band_idx)
                band_start_index += band_set.count


def show_image_with_mask(imageId, maskAlpha=0.4, figsize=(10, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    # print(fig)

    img = stretch_n(M(imageId))
    mask = generate_mask_for_image_and_class(img.shape[:2], imageId)

    tifffile.imshow(img, figure=figsize, subplot=ax)
    ax.imshow(mask, alpha=maskAlpha)
    ax.axis("off")
    plt.show()


def parse_mask_data(df):
    new_df = df.copy()

    num_polygons = []
    img_building_area_ratio = []

    for i in range(len(df)):
        polygons = wkt_loads(df.iloc[i].MultipolygonWKT)
        num_polygons.append(len(polygons))

        imageId = df.iloc[i].ImageId
        img = M(imageId)

        mask = generate_mask_for_image_and_class(img.shape[:2], imageId)
        img_building_area_ratio.append(np.mean(mask))

    new_df["NumPolygons"] = num_polygons
    new_df["BuildingAreaRatio"] = img_building_area_ratio

    return new_df


def _convert_coordinates_to_raster(coords, img_size, xymax):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask

    Xmax, Ymax = xymax
    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
    xf = W1 / Xmax
    yf = H1 / Ymax
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int


def _get_xmax_ymin(grid_sizes_panda, imageId):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
    return (xmax, ymin)


def _get_polygon_list(wkt_list_pandas, imageId, cType):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList


def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list, interior_list


def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    img_mask = np.zeros(raster_img_size, np.uint8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)
    return img_mask


def generate_mask_for_image_and_class(raster_size, imageId, class_type = 1, grid_sizes_panda = GS ,
                                      wkt_list_pandas = DF):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas, imageId, class_type)
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    mask = _plot_mask_from_contours(raster_size, contours, 1)
    return mask


def M(image_id):
    # __author__ = amaia
    # https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
    filename = os.path.join(rootDir / "three_band", '{}.tif'.format(image_id))
    img = tifffile.imread(filename)
    img = np.rollaxis(img, 0, 3)    # change channel
    return img


def M_sixteen_band(image_id, channel="M"):
    filename = os.path.join(rootDir / "sixteen_band", '{}_{}.tif'.format(image_id, channel))
    img = tifffile.imread(filename)
    img = np.rollaxis(img, 0, 3)
    return img


def stretch_n(bands, lower_percent=5, higher_percent=95):
    # https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras
    out = np.zeros_like(bands)
    n = bands.shape[2]
    for i in range(n):
        a = np.min(bands[:, :, i])
        b = np.max(bands[:, :, i])
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    return out.astype(np.float32)


def parse_images(imgDir):
    data = []
    for img_path in imgDir.iterdir():
        imageId = img_path.stem
        img = M(imageId)

        img_data = {
            "ImageId": imageId,
            "height": img.shape[0],
            "width": img.shape[1],
            "depth": img.shape[2],
            "img_channel_max": np.array([img[:, :, channel].max() for channel in range(img.shape[2])]),
            "img_channel_min": np.array([img[:, :, channel].min() for channel in range(img.shape[2])])
        }

        data.append(img_data)

    return pd.DataFrame(data)


def stack(band_paths, out_path):
    """Stack a set of bands into a single file.
    Based on: https://github.com/mapbox/rasterio/issues/1273#issue-295230001

    Parameters
    ----------
    band_paths : list of file paths
        A list with paths to the bands you wish to stack. Bands
        will be stacked in the order given in this list.
    out_path : string
        A path for the output file.
    """
    first_band = rasterio.open(band_paths[0], 'r')
    meta = first_band.meta.copy()

    counts = 0
    for ifile in band_paths:
        first_band = rasterio.open(band_paths[0], 'r')
        with rasterio.open(ifile, 'r') as ff:
            counts += ff.meta['count']
    meta.update(count=counts)

    band_start_index = 1

    with rasterio.open(out_path, 'w', **meta) as ff:
        for img_path in band_paths:
            with rasterio.open(img_path, 'r') as band_set:
                read_band_idx = range(1, 1 + band_set.count)
                bands = band_set.read(read_band_idx,
                                      out_shape=(band_set.count, meta["height"], meta["width"]),
                                      resampling=Resampling.bilinear)
                if bands.ndim != 3:
                    bands = bands[np.newaxis, ...]
                write_band_idx = range(band_start_index, band_start_index + band_set.count)
                ff.write(bands, write_band_idx)
                band_start_index += band_set.count


def preprocess_images(imageIds, dst, threeBandImagesDir):
    for imageId in tqdm(imageIds, leave=False):
        band_paths = [threeBandImagesDir / f"{imageId}.tif"]
        stack(band_paths=band_paths, out_path=dst / f"{imageId}.tif")


def calculate_channel_means_and_stds(imageIds, img_dir, on_normalized=True):
    scaler = StandardScaler()

    for imageId in tqdm(imageIds, leave=False):
        with rasterio.open(img_dir / f"{imageId}.tif", 'r') as img:
            img_np = img.read(range(1, img.count + 1))
            if on_normalized:
                img_np = img_np / 2047
            img_np = np.array([img_np[i, :, :].ravel() for i in range(img_np.shape[0])]).transpose()
            scaler.partial_fit(img_np)

class ImgMinMaxScaler:
    def __init__(self, channel_min, channel_max):
        self.channel_min = channel_min
        self.channel_max = channel_max

    def __call__(self, X):
        return (X - self.channel_min) / (self.channel_max - self.channel_min)


class ImgToTensor:
    def __call__(self, X):
        return torchvision.transforms.functional.to_tensor(X.astype(np.float32))


class ImgResize:
    def __init__(self, dst_size):
        self.dst_size = dst_size

    def __call__(self, X):
        return cv2.resize(X, self.dst_size, interpolation=cv2.INTER_AREA)


class AdversarialDataSet(Dataset):
    def __init__(self, train_img_ids, test_img_ids, transform=None):
        super(AdversarialDataSet, self).__init__()
        self.train_img_ids = train_img_ids
        self.test_img_ids = test_img_ids
        self.transform = transform

    def __len__(self):
        return len(self.train_img_ids) + len(self.test_img_ids)

    def __getitem__(self, index):
        is_test_img = np.random.randint(0, 2)
        img_id = np.random.choice(self.test_img_ids if is_test_img else self.train_img_ids)

        img = M(img_id)

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.as_tensor(is_test_img)


class Cropper:
    def __init__(self, crop_height=300, crop_width=300, n_crops=None):
        """ Creates a cropper.

        Parameters
        ----------
        crop_height : integer
            The height of the crop.
        crop_width : string
            The width of the crop.
        n_crops (optional, default=None) : integer
            If n_crops=None then the maximum number of rectangles of size
            (crop_width, crop_height) will be cut from an image as uniformly as
            possible. If n_crops is an integer then n_crops
            randomly positioned rectangles of size
            (crop_width, crop_height) will be cut from an image.
        """
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.n_crops = n_crops
        self.imageIdToNCropsMap = {}

    def crop(self, imageId, img_root_dir, imgSaveDir, maskSaveDir):
        img_path = img_root_dir / f"{imageId}.tif"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.open(img_path) as img:
                img_height, img_width = img.shape
                mask = generate_mask_for_image_and_class(img.shape, imageId)

                crop_shapes = self.create_crop_rectangles(img_width, img_height)
                self.imageIdToNCropsMap[imageId] = len(crop_shapes)

                for i, crop_rectangle in enumerate(crop_shapes):
                    filename = f"{imageId}_{i}"
                    self._save_cropped_image(img, crop_rectangle,
                                             imgSaveDir / f"{filename}.tif")
                    self._save_cropped_mask(mask, crop_rectangle,
                                            maskSaveDir / f"{filename}.png")

    def crop_all(self, imageIds, img_root_dir, imgSaveDir, maskSaveDir):
        # for imageId in tqdm(imageIds, desc="Cropping images", position=0, leave=False):
        for imageId in imageIds:
            self.crop(imageId, img_root_dir, imgSaveDir, maskSaveDir)

    def create_crop_rectangles(self, img_width, img_height):
        if self.n_crops is not None:
            return self._create_random_crop_rectangles(img_width, img_height)
        return self._create_equidistant_crop_rectangles(img_width, img_height)

    def n_total_created_crops(self):
        return int(np.sum(list(self.imageIdToNCropsMap.values())))

    def _create_random_crop_rectangles(self, img_width, img_height):
        g = GeoSeries([])

        for i in range(self.n_crops):
            while (True):
                crop_x = random.randint(0, img_width - self.crop_width)
                crop_y = random.randint(0, img_height - self.crop_height)
                polygon = self._create_rectangle(crop_x, crop_y,
                                                 self.crop_width,
                                                 self.crop_height)
                if not any(g.intersects(polygon)):
                    break
            g = g.append(GeoSeries([polygon]))
        return g

    def _create_equidistant_crop_rectangles(self, img_width, img_height):
        n_crops_per_row = img_width // self.crop_width
        n_crops_per_col = img_height // self.crop_height

        remaining_x_space = img_width % self.crop_width
        remaining_y_space = img_height % self.crop_height

        x_spacings = np.zeros(n_crops_per_row)
        x_spacings[1:] = remaining_x_space // (n_crops_per_row - 1)

        if remaining_x_space % (n_crops_per_row - 1) != 0:
            x_spacings[-1] += remaining_x_space % (n_crops_per_row - 1)

        y_spacings = np.zeros(n_crops_per_col)
        y_spacings[1:] = remaining_y_space // (n_crops_per_col - 1)

        if remaining_y_space % (n_crops_per_col - 1) != 0:
            y_spacings[-1] += remaining_y_space % (n_crops_per_col - 1)

        polygons = []

        for i in range(n_crops_per_col):
            for j in range(n_crops_per_row):
                crop_x = j * self.crop_width + np.sum(x_spacings[:j + 1])
                crop_y = i * self.crop_height + np.sum(y_spacings[:i + 1])

                polygon = self._create_rectangle(crop_x, crop_y,
                                                 self.crop_width,
                                                 self.crop_height)
                polygons.append(polygon)

        return GeoSeries(polygons)

    @staticmethod
    def _create_rectangle(x, y, w, h):
        return Polygon([(x, y), (x, y + h), (x + w, y + h), (x + w, y)])

    @staticmethod
    def _save_cropped_mask(mask, crop_rectangle, path):
        xmin, ymin, xmax, ymax = (int(coord) for coord in crop_rectangle.bounds)
        crop_mask = mask[ymin:ymax, xmin:xmax]
        crop_mask_img = Image.fromarray(crop_mask)
        crop_mask_img.save(path)

    @staticmethod
    def _save_cropped_image(img, crop_rectangle, path):
        crop_img, _ = mask(img, [crop_rectangle], crop=True)
        with rasterio.open(path, 'w', driver='GTiff', width=crop_img.shape[2],
                           height=crop_img.shape[1], count=crop_img.shape[0],
                           dtype=crop_img.dtype) as img_file:
            img_file.write(crop_img)


class TiffImgDataset(Dataset):
    def __init__(self, img_ids, transform=None):
        super(TiffImgDataset, self).__init__()
        self.img_ids = img_ids
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img = M(self.img_ids[index])

        if self.transform is not None:
            img = self.transform(img)

        return img

class ImgMaskTransformCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class ImgMaskMinMaxScaler:
    def __init__(self, img_mins, img_maxs):
        self.img_mins = img_mins
        self.img_maxs = img_maxs

    def __call__(self, img, mask):
        return (img - self.img_mins) / (self.img_maxs - self.img_mins), mask

class ImgMaskTensorNormalize(torchvision.transforms.Normalize):
    def __call__(self, img, mask):
        return super(ImgMaskTensorNormalize, self).__call__(img), mask

class ImgMaskRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = np.fliplr(img)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask

class ImgMaskRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = np.flipud(img)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask

class ImgMaskToTensor:
    def __call__(self, img, mask):
        return torchvision.transforms.functional.to_tensor(img.astype(np.float32)), torch.as_tensor(np.array(mask), dtype=torch.uint8)