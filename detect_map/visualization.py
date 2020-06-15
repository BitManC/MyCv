from .utils import M
from .utils import M_sixteen_band
from .utils import stretch_n
from .utils import generate_mask_for_image_and_class

import matplotlib.pyplot  as plt
import gc
import tifffile


def show_image(imageId, fig=None, ax=None):
    do_show = False
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        do_show = True
    img = M(imageId)
    img_color_stretched = stretch_n(img)
    tifffile.imshow(img_color_stretched, figure=fig, subplot=ax)
    ax.axis("off")
    if do_show:
        plt.show()


def show_image_channel(imageId, channel="M", fig=None, ax=None):
    do_show = False
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        do_show = True
    img = M_sixteen_band(imageId, channel=channel)
    img_color_stretched = stretch_n(img)
    tifffile.imshow(img_color_stretched, figure=fig, subplot=ax)
    ax.axis("off")
    if do_show:
        plt.show()


def show_images(imageIds, n_max_cols=2, single_img_size=(6, 6)):
    n_images = len(imageIds)
    n_plot_cols = n_max_cols
    n_plot_rows = int((n_images - 1) / n_plot_cols + 1)
    fig, ax = plt.subplots(nrows=n_plot_rows, ncols=n_plot_cols,
                           figsize=(n_plot_cols * single_img_size[0] + 1,
                                    n_plot_rows * single_img_size[1] + 1))

    for i, imageId in enumerate(imageIds):
        img_ax = ax[i // n_plot_cols][i % n_plot_cols]
        show_image(imageId, fig, img_ax)

    plt.tight_layout(pad=0.0, w_pad=0.6, h_pad=0.6)
    plt.show()
    gc.collect()


def show_image_with_mask(imageId, maskAlpha=0.4, figsize=(10, 10)):
    fig, ax = plt.subplots(figsize=figsize)

    img = stretch_n(M(imageId))
    mask = generate_mask_for_image_and_class(img.shape[:2], imageId)

    tifffile.imshow(img, figure=fig, subplot=ax)
    ax.imshow(mask, alpha=maskAlpha)
    ax.axis("off")
    plt.show()