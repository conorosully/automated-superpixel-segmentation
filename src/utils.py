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


def scale_bands(img,satellite="landsat"):
    """Scale bands to 0-1"""
    img = img.astype("float32")
    if satellite == "landsat":
        img = np.clip(img * 0.0000275 - 0.2, 0, 1)
    elif satellite == "sentinel":
        img = np.clip(img * 0.0001, 0, 1)
    return img




# =====================================================================================================
# ========================================  EVALUATION  ===============================================
# =====================================================================================================
