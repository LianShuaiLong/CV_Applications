import numpy as np


from PIL import Image
from PIL.Image import Image as PILImage
from scipy.ndimage import binary_erosion
from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
from pymatting.foreground.estimate_foreground_ml import estimate_foreground_ml
from pymatting.util.util import stack_images
from cv2 import (
    BORDER_DEFAULT,
    MORPH_OPEN,
    GaussianBlur,
    morphologyEx,
    MORPH_ELLIPSE,
    getStructuringElement,
)
import time

def get_time(func):
    def wrap_func(*args,**kwargs):
        s_time = time.time()
        res =func(*args,**kwargs)
        e_time = time.time()
        time_consume = e_time-s_time
        return res,time_consume
    return wrap_func


@get_time
def preprocess_img(img,mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),size=(320, 320)):
    img = img.convert('RGB').resize(size,Image.Resampling.LANCZOS)
    im_ary = np.array(img)
    im_ary = im_ary / np.max(im_ary)

    tmpImg = np.zeros((im_ary.shape[0], im_ary.shape[1], 3))
    tmpImg[:, :, 0] = (im_ary[:, :, 0] - mean[0]) / std[0]
    tmpImg[:, :, 1] = (im_ary[:, :, 1] - mean[1]) / std[1]
    tmpImg[:, :, 2] = (im_ary[:, :, 2] - mean[2]) / std[2]

    tmpImg = tmpImg.transpose((2, 0, 1))
    return tmpImg

@get_time
def postprocess_mask(img,pred):
    ma = np.max(pred)
    mi = np.min(pred)
    pred = (pred - mi) / (ma - mi)
    pred = np.squeeze(pred)

    mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
    mask = mask.resize(img.size, Image.Resampling.LANCZOS)
    return mask

@get_time
def adjust_mask(mask: np.ndarray) -> np.ndarray:
    """
    Post Process the mask for a smooth boundary by applying Morphological Operations
    Research based on paper: https://www.sciencedirect.com/science/article/pii/S2352914821000757
    args:
        mask: Binary Numpy Mask
    """
    kernel = getStructuringElement(MORPH_ELLIPSE, (3, 3))
    mask = morphologyEx(mask, MORPH_OPEN, kernel)
    mask = GaussianBlur(mask, (5, 5), sigmaX=2, sigmaY=2, borderType=BORDER_DEFAULT)
    mask = np.where(mask < 127, 0, 255).astype(np.uint8)  # convert again to binary
    return mask

def naive_cutout(img: PILImage, mask: PILImage) -> PILImage:
    empty = Image.new("RGBA", (img.size), 0)
    cutout = Image.composite(img, empty, mask)
    return cutout

@get_time
def alpha_matting_cutout(
    img: PILImage,
    mask: PILImage,
    foreground_threshold: int,
    background_threshold: int,
    erode_structure_size: int,
) -> PILImage:

    if img.mode == "RGBA" or img.mode == "CMYK":
        img = img.convert("RGB")

    img = np.asarray(img)
    mask = np.asarray(mask)

    is_foreground = mask > foreground_threshold
    is_background = mask < background_threshold

    structure = None
    if erode_structure_size > 0:
        structure = np.ones(
            (erode_structure_size, erode_structure_size), dtype=np.uint8
        )

    is_foreground = binary_erosion(is_foreground, structure=structure)
    is_background = binary_erosion(is_background, structure=structure, border_value=1)

    trimap = np.full(mask.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground] = 255
    trimap[is_background] = 0

    img_normalized = img / 255.0
    trimap_normalized = trimap / 255.0

    alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
    foreground = estimate_foreground_ml(img_normalized, alpha)
    cutout = stack_images(foreground, alpha)

    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)
    cutout = Image.fromarray(cutout)

    return cutout
