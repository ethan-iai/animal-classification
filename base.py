import os
import json
import torch
import numpy as np
import torch.utils.data as data

from PIL import Image
from typing import Any, Callable, List, Optional, Tuple
from torchvision.datasets import VisionDataset

from utils import read_images

NUM_CLASSES = 22
LABELS = [
    "ape", "bear", "bison", "cat", 
    "chicken", "cow", "deer", "dog",
    "dolphin", "duck", "eagle", "fish", 
    "horse", "lion", "lobster", "pig", 
    "rabbit", "shark", "snake", "spider", 
    "turkey", "wolf"
]
LABEL_MAP = {
    0: "ape", 1: "bear", 2: "bison", 3: "cat", 
    4: "chicken", 5: "cow", 6: "deer", 7: "dog",
    8: "dolphin", 9: "duck", 10: "eagle", 11: "fish", 
    12: "horse", 13: "lion", 14: "lobster", 
    15: "pig", 16: "rabbit", 17: "shark", 18: "snake", 
    19: "spider", 20:  "turkey", 21: "wolf"
}

class AnimalDataset(VisionDataset):
    def __init__(
        self, 
        root: str,
        train: bool,
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None
        ) -> None:
            
            super().__init__(root, transform=transform, 
                             target_transform=target_transform)
            self.train = train

            self.data = []
            self.targets = []

            if self.train:
                dir = os.path.join(root, "train")
                for i, dir in enumerate(sorted(dir)):
                    imgs = read_images(dir)
                    self.data.extend(imgs)    
                    self.targets.extand([i, ] * len(imgs))
            
                self.data = np.stack(self.data, axis=0)
            else:
                dir = os.path.join(root, "test")
                imgs = read_images(dir)   # make sure read the images at dict sequence
                self.data.extend(imgs)
                imgs_label_names_map = json.load(
                    os.path.join(root, "test", "test_data.json")
                )
                self.targets = list(map(LABELS.index, 
                                        imgs_label_names_map.values()))


    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
