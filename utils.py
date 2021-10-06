import os
import numpy as np

def read_image(*args, **kwargs):
    # TODO: 
    pass

def read_images(path, dtype, shape=None):
    """[summary]

    Args:
        path ([type]): [description]
        dtype ([type]): [description]
        shape ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    imgs = []
    valid_suffix = \
        set([".jpg", ".gif", ".png", 
            ".tga", ".jpeg", ".ppm"])
    for img in sorted(os.listdir(path)):
        ext = os.path.splitext(img)[-1]
        if ext.lower() not in valid_suffix:
            continue
        # TODO: 
        img = read_image()

        if shape is not None:
            # TODO: reshape img
            pass    

        imgs.append(img)
    
    return imgs


def resize_image(arr, shape):
    """[summary]

    Args:
        arr ([type]): [description]
        shape ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    return np.zeros_like(arr)

