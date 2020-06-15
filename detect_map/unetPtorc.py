import pandas as pd
import numpy as np
import cv2
from shapely.geometry import Polygon
import tifffile
import os
from pathlib import Path
from geopandas import GeoSeries
import rasterio
import rasterio.mask
import random
from PIL import Image
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm.auto import tqdm
from shapely.wkt import loads as wkt_loads
from torch_lr_finder import LRFinder
from collections import defaultdict
from matplotlib.pyplot import plot as plt


rootDir = Path("/Users/yifz/kaggle/unet/dstl")
DF = pd.read_csv(rootDir / 'train_wkt_v4.csv.zip')
GS = pd.read_csv(rootDir / 'grid_sizes.csv.zip', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)


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


def generate_mask_for_image_and_class(raster_size, imageId, class_type=1, grid_sizes_panda=GS, wkt_list_pandas=DF):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas, imageId, class_type)
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    mask = _plot_mask_from_contours(raster_size, contours, 1)
    return mask


def M(image_id):
    # __author__ = amaia
    # "Contrast enhancement", similar to the default when opening an image using QGIS.
    # https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
    filename = os.path.join(rootDir / "three_band", '{}.tif'.format(image_id))
    img = tifffile.imread(filename)
    img = np.rollaxis(img, 0, 3)
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
                    self._save_cropped_image(img, crop_rectangle, imgSaveDir / f"{filename}.tif")
                    self._save_cropped_mask(mask, crop_rectangle, maskSaveDir / f"{filename}.png")

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


class UNetDoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, activation, padding=1):
        super(UNetDoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            activation,
            nn.Dropout2d(p=dropout)
        )

    def forward(self, X):
        return self.conv(X)


class UNetExpansionBlock(nn.Module):
    def __init__(self, in_size, out_size, dropout, activation, padding=1, use_recurrent_block=False):
        super(UNetExpansionBlock, self).__init__()
        if use_recurrent_block:
            self.conv = UNetRRCNNBlock(in_size, out_size, dropout=dropout, padding=padding)
        else:
            self.conv = UNetDoubleConvBlock(in_size, out_size, dropout, activation, padding)

        self.upconv = nn.ConvTranspose2d(out_size, out_size,
                                         kernel_size=2, stride=2)
        self.relu = activation

    def forward(self, X_up, X_down):
        X = torch.cat((X_up, X_down), 1)
        X = self.conv(X)
        X = self.upconv(X)
        X = self.relu(X)
        return X


class UNetAttentionBlock(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(UNetAttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.Psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        W_g_out = self.W_g(g)
        W_x_out = self.W_x(x)

        out = self.relu(W_g_out + W_x_out)
        out = self.Psi(out)

        return x * out


class UNetRecurrentBlock(nn.Module):
    def __init__(self, n_channels, dropout, rec_depth=3, padding=1):
        super(UNetRecurrentBlock, self).__init__()
        self.rec_depth = rec_depth
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=padding),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, X):
        X_next = self.conv(X)
        for i in range(self.rec_depth - 1):
            X_next = self.conv(X + X_next)

        return X_next


class UNetRRCNNBlock(nn.Module):
    def __init__(self, in_size, out_size, dropout, rec_depth=3, padding=1):
        super(UNetRRCNNBlock, self).__init__()
        self.in_conv = nn.Conv2d(in_size, out_size, kernel_size=1)
        self.rec_conv = nn.Sequential(
            UNetRecurrentBlock(out_size, dropout, rec_depth, padding=padding),
            UNetRecurrentBlock(out_size, dropout, rec_depth, padding=padding)
        )

    def forward(self, X):
        X = self.in_conv(X)
        X_next = self.rec_conv(X)

        return X + X_next


class UNet(nn.Module):
    # Basiert auf:
    # https://arxiv.org/abs/1505.04597 (UNet Paper)
    # https://stackoverflow.com/questions/52235520/how-to-use-pnasnet5-as-encoder-in-unet-in-pytorch
    # https://github.com/jaxony/unet-pytorch/blob/master/model.py
    def __init__(self, n_classes=1,
                 input_channels=3,
                 adaptiveInputPadding=True,
                 pretrained_encoder=True,
                 freeze_encoder=True,
                 activation=nn.ReLU(inplace=True),
                 dropout=0.2,
                 use_hypercolumns=False,
                 use_attention=False,
                 use_recurrent_decoder_blocks=False):

        super(UNet, self).__init__()

        self.input_channels = input_channels

        encoder_backbone = torchvision.models.resnet34(pretrained=pretrained_encoder)  # resnet 零件可以自己来做么？
        encoder_backbone.relu = activation
        self.n_classes = n_classes
        self.adaptiveInputPadding = adaptiveInputPadding
        self.use_hypercolumns = use_hypercolumns
        self.use_attention = use_attention

        d1 = []

        if self.input_channels != 3:
            if pretrained_encoder:
                d1.extend([nn.Conv2d(self.input_channels, 3, 1, bias=False), nn.BatchNorm2d(3)])
            else:
                encoder_backbone.conv1 = nn.Conv2d(self.input_channels,
                                                   encoder_backbone.conv1.out_channels,
                                                   kernel_size=7, stride=2, padding=3, bias=False)

        d1.extend([encoder_backbone.conv1,
                   encoder_backbone.bn1,
                   encoder_backbone.relu,
                   encoder_backbone.layer1])

        d1 = nn.Sequential(*d1)
        d2 = encoder_backbone.layer2
        d3 = encoder_backbone.layer3
        d4 = encoder_backbone.layer4

        self.encoder = nn.ModuleList([d1, d2, d3, d4])  # encoder section

        self.input_size_reduction_factor = 2 ** len(self.encoder)

        bottom_channel_nr = self._calculate_bottom_channel_number()

        self.center = UNetDoubleConvBlock(bottom_channel_nr,
                                          bottom_channel_nr,
                                          dropout, activation)

        if self.use_hypercolumns:
            self.hyper_col_convs = nn.ModuleList(
                [nn.Conv2d(bottom_channel_nr, out_channels=3, kernel_size=3, padding=1)])

        if self.use_attention:
            self.attention_layers = nn.ModuleList([])

        layer_in = bottom_channel_nr * 2
        layer_out = bottom_channel_nr // 2

        self.decoder = nn.ModuleList([])

        for i in range(len(self.encoder)):
            if self.use_attention:
                self.attention_layers.append(UNetAttentionBlock(layer_in // 2, layer_in // 2, layer_in // 2))

            self.decoder.append(UNetExpansionBlock(layer_in, layer_out, dropout, activation,
                                                   use_recurrent_block=use_recurrent_decoder_blocks))

            if i != len(self.encoder) - 1:
                if self.use_hypercolumns:
                    self.hyper_col_convs.append(nn.Conv2d(layer_out, out_channels=3, kernel_size=3, padding=1))
                layer_in //= 2
                layer_out //= 2

        hyper_col_num_channels = len(self.hyper_col_convs) * 3 if self.use_hypercolumns else 0

        self.out_conv = nn.Conv2d(layer_out + hyper_col_num_channels, n_classes, 1)

        if freeze_encoder:
            self.freeze_encoder()

    def forward(self, X):
        d_outputs = []

        if self.adaptiveInputPadding:  # input image fe
            X, padding_lr, padding_tb = self._adaptive_padding(X)

        for d in self.encoder:  # input image to each layer
            X = d(X)
            d_outputs.append(X)

        if self.use_hypercolumns:
            u_outputs = []

        X = self.center(X)

        for i, (u, d_out) in enumerate(zip(self.decoder, reversed(d_outputs))):  # up samples
            if self.use_hypercolumns:
                hyper_col_output = F.interpolate(self.hyper_col_convs[i](X),
                                                 scale_factor=2 ** (len(self.decoder) - i),
                                                 mode="bilinear",
                                                 align_corners=False)
                u_outputs.append(hyper_col_output)

            if self.use_attention:
                d_out = self.attention_layers[i](d_out, X)

            X = u(X, d_out)

        if self.use_hypercolumns:
            u_outputs.append(X)
            X = torch.cat(u_outputs, 1)

        X = self.out_conv(X)

        if self.adaptiveInputPadding:
            if padding_lr != (0, 0) or padding_tb != (0, 0):
                X = self._center_crop(X, padding_lr, padding_tb)

        return X

    def freeze_encoder(self, last_layer_to_freeze=None):
        # freeze means keep original parameters
        if last_layer_to_freeze is None:
            last_layer_to_freeze = len(self.encoder) - 1

        for layer_index in range(last_layer_to_freeze + 1):
            for child in self.encoder[layer_index]:
                for parameter in child.parameters():
                    parameter.requires_grad = False

    def unfreeze_encoder(self, first_layer_to_unfreeze=None):
        if first_layer_to_unfreeze is None:
            first_layer_to_unfreeze = 0

        for layer_index in range(first_layer_to_unfreeze, len(self.encoder)):
            for child in self.encoder[layer_index]:
                for parameter in child.parameters():
                    parameter.requires_grad = True

    def encoder_freeze_status(self):
        frozen_status = [all([all([not param.requires_grad for param in child.parameters()])
                              for child in layer.children()]) for layer in self.encoder]

        return ["frozen" if status else "unfrozen" for status in frozen_status]

    def _adaptive_padding(self, X):
        X_height, X_width = X.shape[2:4]

        if X_width % self.input_size_reduction_factor != 0 or X_height % self.input_size_reduction_factor != 0:
            required_padding_left_right = self._calculate_required_adaptive_padding(X_width)
            required_padding_top_bottom = self._calculate_required_adaptive_padding(X_height)

            X = nn.ReflectionPad2d(required_padding_left_right + required_padding_top_bottom)(X)
            return X, required_padding_left_right, required_padding_top_bottom

        return X, (0, 0), (0, 0)

    def _calculate_required_adaptive_padding(self, side_length):
        next_same_output_producing_size = self.input_size_reduction_factor * (
                side_length // self.input_size_reduction_factor + 1)
        required_padding_space = next_same_output_producing_size - side_length

        required_padding = (required_padding_space // 2, required_padding_space // 2 + (required_padding_space % 2))

        return required_padding

    def _center_crop(self, X, lr_offset, tb_offset):
        X_height, X_width = X.shape[2:4]
        return X[:, :, tb_offset[0]: (X_height - tb_offset[1]), lr_offset[0]: (X_width - lr_offset[1])]

    def _calculate_bottom_channel_number(self):
        return [l for l in self.encoder[-1][-1].children() if isinstance(l, nn.Conv2d)][-1].out_channels



class Trainer:

    def __init__(self, model, dl_train, criterion, optimizer,
                 dl_valid=None, dl_test=None, device="cuda",
                 save_dir=None, metrics=None, on_train_val_epoch_finished_callback=None):

        self.model = model
        self.dl_train = dl_train
        self.num_train_samples = len(dl_train.dataset)
        self.criterion = criterion
        self.optimizer = optimizer
        self.dl_valid = dl_valid
        self.num_validation_samples = len(dl_valid.dataset)

        self.dl_test = dl_test
        self.device = device
        self.save_dir = save_dir
        self.metrics = metrics
        self.recorder = MetricRecorder(self.metrics)
        self.latest_lr_finder_result = None
        self.on_train_val_epoch_finished_callback = on_train_val_epoch_finished_callback

    def train(self, num_epochs, max_lr=1e-4, hide_progress=False):
        self.model.to(self.device)
        self.model.train()

        scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=max_lr,
                                                        steps_per_epoch=len(self.dl_train),
                                                        epochs=num_epochs)

        self.recorder.reset()

        for epoch in range(num_epochs, desc="Epochs", disable=hide_progress):
            self._train_one_epoch(epoch, scheduler, hide_progress)

            if self.dl_valid is not None:
                self._validate(epoch, hide_progress)

            if self.on_train_val_epoch_finished_callback is not None:
                self.on_train_val_epoch_finished_callback(epoch)

        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _train_one_epoch(self, epoch, scheduler, hide_progress):
        self.model.train()

        for img_batch, mask_batch in tqdm(self.dl_train, desc=f"Epoch {epoch}", leave=False,
                                          disable=hide_progress, position=0):
            img_batch, mask_batch = img_batch.to(self.device), mask_batch.to(self.device)

            self.optimizer.zero_grad()

            model_output = self.model(img_batch)

            loss = self.criterion(model_output, mask_batch)

            with torch.no_grad():
                self.recorder.update_record_on_batch_end(epoch, loss.item(), mask_batch,
                                                         model_output.squeeze(),
                                                         img_batch.size(0),
                                                         self.num_train_samples)
            loss.backward()
            self.optimizer.step()
            scheduler.step()

        with torch.no_grad():
            self.recorder.finalize_record_on_epoch_end()

        tqdm.write(self.recorder.get_latest_epoch_message(training=True))

    def _validate(self, epoch, hide_progress):
        assert self.dl_valid is not None
        self.model.eval()

        with torch.no_grad():
            for img_batch, mask_batch in tqdm(self.dl_valid, desc="Validating",
                                              disable=hide_progress, leave=False, position=0):
                img_batch, mask_batch = img_batch.to(self.device), mask_batch.to(self.device)
                model_output = self.model(img_batch)
                loss = self.criterion(model_output, mask_batch)

                self.recorder.update_record_on_batch_end(epoch, loss.item(), mask_batch,
                                                         model_output.squeeze(),
                                                         img_batch.size(0),
                                                         self.num_validation_samples,
                                                         training=False)
            self.recorder.finalize_record_on_epoch_end(training=False)
            tqdm.write(self.recorder.get_latest_epoch_message(training=False))

    def lr_range_test(self, val_loss=False):
        lr_finder = LRFinder(self.model, self.optimizer, self.criterion, device=self.device)

        val_loader = self.dl_valid if val_loss else None

        lr_finder.range_test(self.dl_train, val_loader=val_loader, end_lr=100,
                             num_iter=100, step_mode="exp")

        lr_finder.plot()
        lr_finder.reset()
        self.latest_lr_finder_result = lr_finder

    def predict(self, image_input, threshold=0.5, image_preprocessing_cb=None):
        self.model.eval()

        if image_preprocessing_cb is not None:
            image_input = image_preprocessing_cb(image_input)

        with torch.no_grad():
            image_input = image_input.to(self.device)
            model_out = self.model(image_input)

        return model_out


class MetricRecorder:
    def __init__(self, metrics):
        self.data = {"training": [], "validation": []}
        self.metrics = metrics

    def reset(self):
        self.data = {"training": [], "validation": []}

        for metric in self.metrics:
            metric.reset()

    def update_record_on_batch_end(self, epoch, loss, actual, prediction,
                                   n_batch_samples, n_total_samples,
                                   threshold=0.5, training=True):
        records = self.data["training" if training else "validation"]

        if epoch >= len(records):
            record = defaultdict(float)
            record["epoch"] = epoch
            records.append(record)
        else:
            record = records[epoch]

        bs_ratio = n_batch_samples / n_total_samples

        record["loss"] += loss * bs_ratio

        if self.metrics is not None:
            pred_proba = (torch.sigmoid(prediction) > threshold).type(actual.dtype)
            for metric in self.metrics:
                metric.on_batch_end(actual, pred_proba, bs_ratio)

    def finalize_record_on_epoch_end(self, training=True):
        record = self.data["training" if training else "validation"][-1]

        metrics_dict = {}
        for metric in self.metrics:
            metrics_dict.update(metric.on_epoch_end())

        record.update(metrics_dict)

    def get_records_dataframe(self, training=True):
        return pd.DataFrame(self.data["training" if training else "validation"]).set_index("epoch")

    def get_latest_epoch_message(self, training=True):
        record_type = "Train" if training else "Valid"
        record = self.data["training" if training else "validation"][-1]

        message = "Epoch {0} - {1}: loss={2:.6f}".format(record["epoch"], record_type, record["loss"])
        metric_message = ', '.join(f"{key}={val:.4f}" for key, val in record.items()
                                   if key is not "epoch" and key is not "loss")
        if metric_message:
            message += ", " + metric_message
        return message

    def show_train_val_metric_curve(self, metric="loss", figsize=(10, 8)):
        fig, ax = plt.subplots(figsize=figsize)

        train_pd = self.get_records_dataframe()
        val_pd = self.get_records_dataframe(training=False)

        train_pd.plot(y=metric, ax=ax, label=f"train-{metric}")
        val_pd.plot(y=metric, ax=ax, label=f"val-{metric}")
        plt.show()


class MetricCallback:
    def on_batch_end(self, actual, prediction, bs_ratio):
        pass

    def on_epoch_end(self):
        pass

    def reset(self):
        pass


class AccuracyMetric(MetricCallback):
    name = "acc"

    def __init__(self):
        self.reset()

    def reset(self):
        self.acc = torch.tensor(0.0)

    def on_batch_end(self, actual, prediction, bs_ratio):
        self.acc += (prediction == actual).float().mean() * bs_ratio

    def on_epoch_end(self):
        result = {self.name: self.acc.item()}
        self.reset()

        return result


class PrecisionRecallF1Metric(MetricCallback):
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = torch.tensor(0)
        self.tn = torch.tensor(0)
        self.fp = torch.tensor(0)
        self.fn = torch.tensor(0)

    def on_batch_end(self, actual, prediction, bs_ratio):
        self.tp += (actual * prediction).sum()
        self.tn += ((1 - actual) * (1 - prediction)).sum()
        self.fp += ((1 - actual) * prediction).sum()
        self.fn += (actual * (1 - prediction)).sum()

    def on_epoch_end(self):
        sum_tp_fp = self.tp + self.fp
        precision = self.tp.float() / sum_tp_fp.float() if sum_tp_fp is not 0 else 0

        sum_tp_fn = self.tp + self.fn
        recall = self.tp.float() / sum_tp_fn.float() if sum_tp_fn is not 0 else 0

        sum_precison_recall = precision + recall
        f1 = 2.0 * precision * recall / sum_precison_recall.float() if sum_precison_recall is not 0 else 0

        self.reset()

        return {"prec": precision.item(), "recall": recall.item(), "f1": f1.item()}


class IOUMetric(MetricCallback):
    def __init__(self, smooth=1e-6):
        self.smooth = smooth
        self.reset()

    def reset(self):
        self.intersection = torch.tensor(0)
        self.union = torch.tensor(0)

    def on_batch_end(self, actual, prediction, bs_ratio):
        self.intersection += (actual & prediction).sum()
        self.union += (actual | prediction).sum()

    def on_epoch_end(self):
        result = (self.intersection.float() + self.smooth) / (self.union.float() + self.smooth)
        self.reset()
        return {"iou": result.item()}
