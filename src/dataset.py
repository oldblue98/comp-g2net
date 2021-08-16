import numpy as np
import torch
from torch.utils.data import Dataset

from src.augmentation import *
from src.utils import *


class ImageDataset(Dataset):
    def __init__(self, train_df, transforms=None):

        self.image_paths = train_df["image_path"].values
        self.labels = train_df["label"].values
        self.augmentations = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        target = self.labels[index]

        image = np.load(image_path)
        image = np.vstack(image[::2, ...]).astype(float)
        # image = cv2.resize(image, (image.shape[0]//2, image.shape[0]//2))
        image = image.astype("f")[..., np.newaxis]
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        return image, torch.tensor(target)
