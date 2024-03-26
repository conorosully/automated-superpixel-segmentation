# Class containing utility functions for the project

import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import cv2
import rasterio as rio
import spyndex
import xarray as xr

base_path = "../data/"
save_path = "/Users/conorosullivan/Google Drive/My Drive/UCD/research/Journal Paper 1 - Irish coastline with landsat/figures"


def save_fig(fig, name):
    """Save figure to figures folder"""
    fig.savefig(save_path + f"/{name}.png", dpi=300, bbox_inches="tight")


def get_bands(ID):
    # Load all bands for a given landsat image

    if ID[0:4] in ["LT05", "LE07"]:
        # Landsat 5 and 7
        B1 = tiff.imread(base_path + "{}/{}_SR_B1.TIF".format(ID, ID))  # Blue
        B2 = tiff.imread(base_path + "{}/{}_SR_B2.TIF".format(ID, ID))  # Green
        B3 = tiff.imread(base_path + "{}/{}_SR_B3.TIF".format(ID, ID))  # Red
        B4 = tiff.imread(base_path + "{}/{}_SR_B4.TIF".format(ID, ID))  # NIR
        B5 = tiff.imread(base_path + "{}/{}_SR_B5.TIF".format(ID, ID))  # SWIR1
        B7 = tiff.imread(base_path + "{}/{}_SR_B7.TIF".format(ID, ID))  # SWIR2
        B6 = tiff.imread(base_path + "{}/{}_ST_B6.TIF".format(ID, ID))  # Thermal

        bands = np.dstack((B1, B2, B3, B4, B5, B7, B6))

    else:
        # Landsat 8 and 9
        B2 = tiff.imread(base_path + "{}/{}_SR_B2.TIF".format(ID, ID))  # Blue
        B3 = tiff.imread(base_path + "{}/{}_SR_B3.TIF".format(ID, ID))  # Green
        B4 = tiff.imread(base_path + "{}/{}_SR_B4.TIF".format(ID, ID))  # Red
        B5 = tiff.imread(base_path + "{}/{}_SR_B5.TIF".format(ID, ID))  # NIR
        B6 = tiff.imread(base_path + "{}/{}_SR_B6.TIF".format(ID, ID))  # SWIR1
        B7 = tiff.imread(base_path + "{}/{}_SR_B7.TIF".format(ID, ID))  # SWIR2
        B10 = tiff.imread(base_path + "{}/{}_ST_B10.TIF".format(ID, ID))  # Thermal 1

        bands = np.dstack((B2, B3, B4, B5, B6, B7, B10))

    return bands


def get_mask(id):
    # Load the coastline mask for a given landsat image

    mask_path = base_path + f"clean masks/{id}.npy"
    mask = np.load(mask_path)
    mask = mask.astype("uint8")

    return mask


def get_stack(id, path="stacked"):
    """Load the stacked bands and mask for a given landsat image"""
    stack_path = base_path + f"{path}/{id}.npy"
    stack = np.load(stack_path)

    return stack


def get_geolocation(id):
    """Load raster used for pixel geolocation"""

    path = base_path + f"raw scenes/{id}/{id}_SR_B3.TIF"
    geolocation = rio.open(path)

    return geolocation


def get_QA(id):
    """Load landsat QA band"""

    path = base_path + f"raw scenes/{id}/{id}_QA_PIXEL.TIF"
    QA = tiff.imread(path)
    QA = np.array(QA)

    return QA


def display_bands(bands, mask=False):
    """Visualize all 7 SR bands and label"""

    band_names = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "Thermal"]
    n = 7

    if mask:
        n = 9
        band_names = [
            "Blue",
            "Green",
            "Red",
            "NIR",
            "SWIR1",
            "SWIR2",
            "Thermal",
            "QA",
            "Mask",
        ]

    fig, axs = plt.subplots(1, n, figsize=(20, 5))

    for i in range(n):
        axs[i].imshow(bands[:, :, i], cmap="gray")
        axs[i].set_title(band_names[i])
        axs[i].axis("off")


def display_scene(id, infrared=False, resize=False):
    """Visualize the RGB image and mask for a given scene"""
    RGB = get_rgb(id, infrared)
    mask = get_mask(id)

    if resize:
        # make image 10x smaller
        RGB = cv2.resize(RGB, (0, 0), fx=0.1, fy=0.1)
        mask = cv2.resize(mask, (0, 0), fx=0.1, fy=0.1)

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].imshow(RGB)
    axs[0].set_title(id, fontsize=18)
    axs[0].axis("off")
    axs[1].imshow(mask, cmap="gray")
    axs[1].axis("off")


def rgb_from_stack(stack, infrared=False, contrast=0.2):
    """Convert a stacked array of bands to RGB"""
    if infrared:
        rgb = stack[:, :, [3, 1, 0]]
    else:
        rgb = stack[:, :, [2, 1, 0]]

    rgb = rgb.astype(np.float32)
    rgb = np.clip(rgb * 0.0000275 - 0.2, 0, 1)
    rgb = np.clip(rgb, 0, contrast) / contrast

    return rgb


def get_rgb(id, infrared=False):
    # Load Blue (B2), Green (B3) and NIR (B5) bands

    save_path = base_path + f"raw scenes/{id}/{id}_SR_"

    if id[0:4] in ["LT05", "LE07"]:
        # Landsat 5 and 7
        if infrared:
            R = tiff.imread(save_path + "B4.TIF")
        else:
            R = tiff.imread(save_path + "B3.TIF")
        G = tiff.imread(save_path + "B2.TIF")
        B = tiff.imread(save_path + "B1.TIF")
    else:
        # Landsat 8 and 9
        if infrared:
            R = tiff.imread(save_path + "B5.TIF")
        else:
            R = tiff.imread(save_path + "B4.TIF")
        G = tiff.imread(save_path + "B3.TIF")
        B = tiff.imread(save_path + "B2.TIF")

    # Stack and scale bands
    RGB = np.dstack((R, G, B))
    RGB = np.clip(RGB * 0.0000275 - 0.2, 0, 1)

    # Clip to enhance contrast
    RGB = np.clip(RGB, 0, 0.2) / 0.2

    return RGB


def scale_bands(bands):
    """Scale bands to 0-1"""
    bands = bands.astype("float32")
    bands = np.clip(bands * 0.0000275 - 0.2, 0, 1)

    return bands


def trim_RGB(RGB):
    """Trim the edges of the image for label studio"""

    img = RGB.copy()

    y, x, z = img.shape

    if y * x > 67000000:
        print("TRIM")
        trim = 20
        img = img[trim : y - trim, trim : x - (trim + 100), :]

    return img


def untrim_label(label):
    """Add the black edges back to the label"""
    trim = 20
    label = np.pad(
        label, ((trim, trim), (trim, trim + 100)), "constant", constant_values=0
    )

    return label


def edge_from_mask(mask):
    """Get edge map from mask"""

    dy, dx = np.gradient(mask)
    grad = np.abs(dx) + np.abs(dy)
    edge = np.array([grad > 0])[0]
    edge = edge.astype(np.uint8)

    return edge


def get_cloud_mask(val, type="cloud"):
    """Get mask for a specific cover type"""

    # convert to binary
    bin_ = "{0:016b}".format(val)

    # reverse string
    str_bin = str(bin_)[::-1]

    # get bit for cover type
    bits = {"cloud": 3, "shadow": 4, "dilated_cloud": 1, "cirrus": 2}
    bit = str_bin[bits[type]]

    if bit == "1":
        return 0  # cover
    else:
        return 1  # no cover


# MODELLING


def get_index(bands, index="NDWI"):
    channels = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "Thermal"]
    standards = ["B", "G", "R", "N", "S1", "S2", "T"]

    """Add indices to image"""

    img = bands.copy()

    img = img[:, :, [0, 1, 2, 3, 4, 5, 6]]
    img = scale_bands(img)

    da = xr.DataArray(img, dims=("x", "y", "band"), coords={"band": channels})

    params = {
        standard: da.sel(band=channel) for standard, channel in zip(standards, channels)
    }

    idx = spyndex.computeIndex(index=[index], params=params)

    idx = np.array(idx)

    # img = np.array(np.vstack((img,idx)))

    return idx


# =====================================================================================================
# ========================================  EVALUATION  ===============================================
# =====================================================================================================
