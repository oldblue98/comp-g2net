import sys
sys.path.append('nnAudio/')

import numpy as np
import torch
from torch.utils.data import Dataset

from src.augmentation import *
from src.utils import *

from nnAudio.Spectrogram import CQT1992v2


class ImageDataset(Dataset):
    def __init__(self, train_df, transforms=None, image_type="spatial"):

        self.image_paths = train_df["image_path"].values
        self.labels = train_df["label"].values
        self.augmentations = transforms
        self.image_type = image_type

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        target = self.labels[index]

        image = np.load(image_path)
        image = self.apply_qtransform(image, self.image_type)
        # image = cv2.resize(image, (image.shape[0]//2, image.shape[0]//2))
        image = image.squeeze().numpy()
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        return image, torch.tensor(target)

    def apply_qtransform(self, waves, image_type, transform=CQT1992v2(sr=2048, fmin=20, fmax=1024, hop_length=64)):
        if image_type == "spatial":
            waves = np.hstack(waves)
            waves = waves / np.max(waves)
            waves = torch.from_numpy(waves).float()
            image = transform(waves)
        elif image_type == "channel":
            image = np.concatenate([transform(torch.from_numpy(waves[i]/np.max(waves)).float()) for i in range(len(waves))], axis=0)
            # image = image.transpose(1, 2, 0)
        else:
            raise Exception("image_type is not defined")
        return image
