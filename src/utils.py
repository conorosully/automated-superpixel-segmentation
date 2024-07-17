# Class containing utility functions for the project

#import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import cv2
#import rasterio as rio
import spyndex
import xarray as xr

from skimage.segmentation import mark_boundaries

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed

from skimage.filters import threshold_otsu

base_path = "../data/"
save_path = "/Users/conorosullivan/Google Drive/My Drive/UCD/research/Journal Paper 2 - superpixels/figures/"

# dictionary of band names for each satellite
band_dic = {"sentinel":{"coastal":0,"blue":1,"green":2,"red":3,"rededge1":4,"rededge2":5,"rededge3":6,"nir":7,"narrownir":8,"watervapour":9,"swir1":10,"swir2":11},
         "landsat":{"blue":0,"green":1,"red":2,"nir":3,"swir1":4,"swir2":5,"thermal":6}}


def save_fig(fig, name):
    """Save figure to figures folder"""
    fig.savefig(save_path + f"/{name}.png", dpi=300, bbox_inches="tight")


def display_bands(img, satellite="landsat"):
    """Visualize all SR bands of a satellite image."""

    if satellite == "landsat":
        n = img.shape[2]
        band_names = ["Blue","Green","Red","NIR","SWIR1","SWIR2","Thermal"]
        if n == 8:
            band_names.append("Mask")
        else:
            # Bands for LICS test set 
            band_names.extend(["QA","Train","Mask","Edge"])

    elif satellite == "sentinel":
        n = 13
        band_names = [ "Coastal","Blue","Green","Red","Red Edge 1","Red Edge 2","Red Edge 3","NIR","Red Edge 4","Water Vapour","SWIR1","SWIR2","Mask"]

    fig, axs = plt.subplots(1, n, figsize=(20, 5))

    for i in range(n):
        if np.unique(img[:, :, i]).size == 1:
            axs[i].imshow(img[:, :, i], cmap="gray", vmin=0, vmax=1)
        else:
            axs[i].imshow(img[:, :, i], cmap="gray")
        axs[i].set_title(band_names[i])
        axs[i].axis("off")


def scale_bands(img,satellite="landsat"):
    """Scale bands to 0-1"""
    img = img.astype("float32")
    if satellite == "landsat":
        img = np.clip(img * 0.0000275 - 0.2, 0, 1)
    elif satellite == "sentinel":
        img = np.clip(img/10000, 0, 1)
    return img

def edge_from_mask(mask):
    """Get edge map from mask"""

    dy, dx = np.gradient(mask)
    grad = np.abs(dx) + np.abs(dy)
    edge = np.array([grad > 0])[0]
    edge = edge.astype(np.uint8)

    return edge


def get_rgb(img, bands=['red','green',"blue"],satellite ='landsat', contrast=1):
    """Convert a stacked array of bands to RGB"""

    r = band_dic[satellite][bands[0]]
    g = band_dic[satellite][bands[1]]
    b = band_dic[satellite][bands[2]]

    rgb = img[:,:, [r,g,b]]
    rgb = rgb.astype(np.float32)
    rgb = scale_bands(rgb, satellite)
    rgb = np.clip(rgb, 0, contrast) / contrast

    return rgb

def histogram_equalization(img):
    """Apply histogram equalization to an image"""

    # Rescale the image to 0-255 and convert to uint8
    if img.max() <= 1:
        img_rescaled = (img * 255).astype(np.uint8)

    # Split the rescaled image into its respective channels (R, G, B)
    R, G, B = cv2.split(img_rescaled)

    # Apply histogram equalization to each channel
    R = cv2.equalizeHist(R)
    G = cv2.equalizeHist(G)
    B = cv2.equalizeHist(B)

    # Merge the equalized channels back together
    img_equalised = cv2.merge((R,G,B))
    
    return img_equalised

def histogram_equalization_luminance(img):
    """Apply histogram equalization to the luminance of an image"""
    
    # Rescale the image to 0-255 if necessary and convert to uint8
    if img.max() <= 1:
        img_rescaled = (img * 255).astype(np.uint8)
    else:
        img_rescaled = img.astype(np.uint8)
    
    # Convert the image from RGB to YCrCb color space
    img_YCrCb = cv2.cvtColor(img_rescaled, cv2.COLOR_RGB2YCrCb)
    
    # Split the image into its respective channels (Y, Cr, Cb)
    Y, Cr, Cb = cv2.split(img_YCrCb)
    
    # Apply histogram equalization to the luminance channel (Y)
    Y_equalized = cv2.equalizeHist(Y)
    
    # Merge the channels back together
    img_YCrCb_equalized = cv2.merge((Y_equalized, Cr, Cb))
    
    # Convert the image back to RGB color space
    img_equalized = cv2.cvtColor(img_YCrCb_equalized, cv2.COLOR_YCrCb2RGB)
    
    return img_equalized


def get_index(bands, index="MNDWI",satellite='sentinel'):

    """Add indices to image"""

    if satellite == 'landsat':

        channels = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", "Thermal"]
        standards = ["B", "G", "R", "N", "S1", "S2", "T"]

        img = bands.copy()
        img = img[:, :, [0, 1, 2, 3, 4, 5, 6]]

    elif satellite == 'sentinel':

        channels = ["Coastal", "Blue", "Green", "Red", "RedEdge1", "RedEdge2", "RedEdge3", "NIR", "NarrowNIR", "WaterVapour", "SWIR1", "SWIR2"]
        standards = ["A","B", "G", "R", "RE1", "RE2", "RE3", "N", "NN", "WV", "S1", "S2"]

        img = bands.copy()
        img = img[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
        
    img = img.astype(np.float32)
    img = scale_bands(img,satellite=satellite)

    da = xr.DataArray(img, dims=("x", "y", "band"), coords={"band": channels})

    params = {
        standard: da.sel(band=channel) for standard, channel in zip(standards, channels)
    }

    idx = spyndex.computeIndex(index=[index], params=params)

    idx = np.array(idx)

    return idx

def get_segments(input_image, method='slic', **kwargs):
    # Copy the input image to avoid modifying the original
    img = input_image.copy()

    # Define a dictionary to map method names to their respective function calls
    segmentation_methods = {
        'felzenszwalb': lambda: felzenszwalb(img, **kwargs),
        'slic': lambda: slic(img, **kwargs),
        'quickshift': lambda: quickshift(img, **kwargs),
        'watershed': lambda: watershed(sobel(rgb2gray(img)), **kwargs),
    }

    # Validate the method
    if method not in segmentation_methods:
        raise ValueError(f"Unsupported segmentation method: {method}. Choose from {list(segmentation_methods.keys())}.")

    # Try to execute the selected method, handle potential errors gracefully
    try:
        segments = segmentation_methods[method]()
    except Exception as e:
        raise ValueError(f"An error occurred during segmentation with {method}: {str(e)}")

    return segments

def get_threshold(index,threshold=-1):
    """Get index threshold prediction using Otsu's method or a specified threshold value."""
    # Ensure the image is in the correct format (float or int) if necessary
    # index = index.astype('uint8') # Uncomment and adjust if needed
    
    # Apply simple threshold to check for binary images
    thresholded_img = index.copy()
    thresholded_img[index >= 0] = 1
    thresholded_img[index < 0] = 0

    if np.unique(thresholded_img).size == 1:
        return thresholded_img

    # Calculate Otsu's threshold
    index = np.nan_to_num(index, nan=0)
    if threshold == -1:
        otsu_threshold = threshold_otsu(index)
    else:
        otsu_threshold = threshold

    # Apply threshold
    thresholded_img = index.copy()
    thresholded_img[index >= otsu_threshold] = 1
    thresholded_img[index < otsu_threshold] = 0
    
    return thresholded_img

def enhance_contrast(img, contrast):
    
    """Enhance the contrast of an image using 3 approaches:
    - contrast = 'hist': apply histogram equalization
    - contrast = 'luminance': apply histogram equalization to the luminance channel
    - contrast = float: rescale the image to the specified value"""

    processed_img = img.copy()
    if contrast == 'hist':
        processed_img = histogram_equalization(processed_img)
    elif contrast == 'luminance':
        processed_img = histogram_equalization_luminance(processed_img)
    else:
        processed_img = processed_img.astype(np.float32)
        processed_img = np.clip(processed_img, 0, contrast) / contrast
    return processed_img


# replace segments with mode prediction
def get_superpixel_mask(segments, thresholded_img):
    """Replace segments with mode prediction"""
    new_segments = np.zeros(segments.shape)
    for i in np.unique(segments):
        mask = segments == i
        mode = np.round(np.mean(thresholded_img[mask]))
        new_segments[mask] = mode

    return new_segments

def get_mask_from_bands(all_bands,satellite,display = False,rgb_bands = ["nir","green","blue"], index_name="MNDWI", threshold=-1,method='slic', **kwargs):
    """Get mask from bands using index and segmentation method"""
    # Get index
    all_bands_ = all_bands.copy()
    img = get_rgb(all_bands_,bands=rgb_bands,satellite=satellite)
    
    index = get_index(all_bands,satellite=satellite,index=index_name)
    segments = get_segments(img, method=method, **kwargs)
    threshold = get_threshold(index,threshold=threshold)
    mask = get_superpixel_mask(segments,threshold)

    if display:
        fig, ax = plt.subplots(1, 6, figsize=(15, 5))

        ax[0].imshow(img)
        ax[0].set_title('Original RGB')
        img_processed = histogram_equalization_luminance(img)
        ax[1].imshow(img_processed)
        ax[1].set_title('Processed RGB')

        ax[2].imshow(mark_boundaries(img, segments))
        ax[2].set_title('Segmentation')

        ax[3].imshow(index,cmap='gray')
        ax[3].set_title('{}'.format(index_name))

        if np.unique(threshold).size == 1:
            ax[4].imshow(threshold,cmap='gray',vmin=0, vmax=1)
        else:
            ax[4].imshow(threshold,cmap='gray')
        ax[4].set_title('Threshold')

        if np.unique(mask).size == 1:
            ax[5].imshow(mask, cmap="gray", vmin=0, vmax=1)
        else:
            ax[5].imshow(mask, cmap="gray")

        ax[5].set_title('Mask')

        for a in ax:
            a.axis('off')
    
    return mask

# =====================================================================================================
# ========================================  EVALUATION  ===============================================
# =====================================================================================================
