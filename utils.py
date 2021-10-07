import os
import numpy as np

from PIL import Image

def read_image(path, shape=None, resample=None):
    """ read single image

    Args:
        path ([type]): [description]
        shape ([type], optional): [description]. Defaults to None.
        resample ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    with Image.open(path) as img:
        if shape is not None:
            img = img.resize(shape, resample)
    return np.asarray(img)

def read_images(path, dtype, shape=None, resample=None):
    """ read images from directory path

    Args:
        path ([type]): [description]
        dtype ([type]): [description]
        shape ([type], optional): [description]. Defaults to None.
        resample ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    imgs = []
    valid_suffix = \
        set([".jpg", ".gif", ".png", 
            ".tga", ".jpeg", ".ppm"])
    for img_path in sorted(os.listdir(path)):
        ext = os.path.splitext(img_path)[-1]
        if ext.lower() not in valid_suffix:
            continue
        img = read_image(img_path, shape, resample)
        imgs.append(img)
    imgs = np.stack(imgs, axis=0)

    return imgs