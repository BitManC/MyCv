import warnings
import rasterio
import rasterio.mask
from .utils import generate_mask_for_image_and_class
import tqdm
import numpy as np
from geopandas import GeoSeries
import random
from shapely.geometry import Polygon
from PIL import Image


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
        for imageId in tqdm(imageIds, desc="Cropping images", position=0, leave=False):
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
        crop_img, _ = rasterio.mask.mask(img, [crop_rectangle], crop=True)
        with rasterio.open(path, 'w', driver='GTiff', width=crop_img.shape[2],
                           height=crop_img.shape[1], count=crop_img.shape[0],
                           dtype=crop_img.dtype) as img_file:
            img_file.write(crop_img)

