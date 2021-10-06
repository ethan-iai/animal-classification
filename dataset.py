import os
import torch

from utils import read_images

NUM_CLASSES = 24
LABEL_MAP = {
    0: "ape", 1: "bear", 2: "bison", 3: "cat", 
    4: "chicken", 5: "cow", 6: "deer", 7: "dog",
    9: "dolphin", 10: "duck", 11: "eagle", 12: "fish", 
    13: "horse", 14: "lion", 15: "lobster", 
    16: "pig", 17: "rabbit", 19: "shark", 20: "snake", 
    21: "spider", 22:  "turkey", 23: "wolf"
}

class AnimalDataset(torch.utils.data.Dataset):
    """ 
    customized animal dataset 
    x scale from (0, 1) 
    """

    def __init__(self, path, 
                 x_shape, y_shape, 
                 x_dtype, y_dtype) -> None:
        _xs = []
        _ys = []
        for i, dir in enumerate(sorted(os.listdir(path))):
            imgs = read_images(dir, x_dtype, x_shape)
            _xs.extend(imgs)    
            _ys.extand([i, ] * len(imgs))

    def __len__(self):
        assert(len(self._xs) == len(self._ys))
        return len(self._xs)

    def __getitem__(self, index):
        return self._xs[index], self._ys[index]