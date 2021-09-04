import sys
sys.path.append('nnAudio/')

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.fft import fft, rfft, ifft

from src.augmentation import *
from src.utils import *

from nnAudio.Spectrogram import CQT1992v2


class ImageDataset(Dataset):
    def __init__(self, train_df, qtransform, transforms=None, image_type="spatial", whiten=True):

        self.image_paths = train_df["image_path"].values
        self.labels = train_df["label"].values
        self.augmentations = transforms
        self.image_type = image_type
        self.transform = CQT1992v2(**qtransform)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        target = self.labels[index]

        signal = np.load(image_path)
        if self.whiten:
            signal = self.whiten(signal)
        image = self.apply_qtransform(signal, self.image_type)
        # print("before",image.shape)
        # image = cv2.resize(image, (image.shape[0]//2, image.shape[0]//2))
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        # print("after",image.shape)
        return image, torch.tensor(target)

    def apply_qtransform(self, waves, image_type):
        transform = self.transform
        if image_type == "spatial":
            waves = np.hstack(waves)
            waves = waves / np.max(waves)
            waves = torch.from_numpy(waves).float()
            image = transform(waves)
            image = image.squeeze().numpy()
        elif image_type == "channel":
            image = np.concatenate([transform(torch.from_numpy(waves[i]/np.max(waves)).float()) for i in range(len(waves))], axis=0)
            image = image.transpose(1, 2, 0)
        else:
            raise Exception("image_type is not defined")
        return image

    def whiten(self, signal):
        hann = torch.hann_window(len(signal), periodic=True, dtype=float)
        spec = fft(torch.from_numpy(signal).float()* hann)
        mag = torch.sqrt(torch.real(spec*torch.conj(spec))) 

        return torch.real(ifft(spec/mag)).numpy() * np.sqrt(len(signal)/2)