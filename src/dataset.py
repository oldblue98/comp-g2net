import sys
sys.path.append('nnAudio/')

import numpy as np
import torch
from torch.utils.data import Dataset

from src.augmentation import *
from src.utils import *

from nnAudio.Spectrogram import CQT1992v2


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
        image = self.apply_qtransform(image)
        # image = cv2.resize(image, (image.shape[0]//2, image.shape[0]//2))
        image = image.squeeze().numpy()
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        return image, torch.tensor(target)

    def apply_qtransform(self, waves, transform=CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=64)):
        waves = np.hstack(waves)
        waves = waves / np.max(waves)
        waves = torch.from_numpy(waves).float()
        image = transform(waves)
        return image
