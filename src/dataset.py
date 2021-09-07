import sys
sys.path.append('nnAudio/')

import numpy as np
from scipy import signal

import torch
from torch.utils.data import Dataset
from torch.fft import fft, rfft, ifft

from src.augmentation import *
from src.utils import *

from nnAudio.Spectrogram import CQT1992v2


class ImageDataset(Dataset):
    def __init__(self, train_df, qtransform, transforms=None, image_type="spatial", use_whiten=False):

        self.image_paths = train_df["image_path"].values
        self.labels = train_df["label"].values
        self.augmentations = transforms
        self.image_type = image_type
        self.transform = CQT1992v2(**qtransform)
        self.use_whiten = use_whiten

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        target = self.labels[index]

        signal = np.load(image_path)
        for j in range(len(signal)):
            signal[j] = self.bandpass(signal[j], 2048)
        if self.use_whiten:
            signal = self.whiten(signal)

        if self.image_type == "spatial":
            image = torch.cat([self.apply_qtransform(signal[j]) for j in range(len(signal))], dim=2)
            image = image.transpose(1, 2, 0)
        elif self.image_type == "channel":
            image = torch.cat([self.apply_qtransform(signal[j]) for j in range(len(signal))], dim=0)
            image = image.transpose(1, 2, 0)
        else:
            raise Exception("image_type is not defined")

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        # print("after",image.shape)
        return image, torch.tensor(target)

    def apply_qtransform(self, waves):
        waves = torch.from_numpy(waves).float()
        image = self.transform(waves)
        return image

    def whiten(self, signal):
        hann = torch.hann_window(len(signal), periodic=True, dtype=float)
        spec = fft(torch.from_numpy(signal).float()* hann)
        mag = torch.sqrt(torch.real(spec*torch.conj(spec))) 

        return torch.real(ifft(spec/mag)).numpy() * np.sqrt(len(signal)/2)

    def bandpass(self, x, fs, fmin=20, fmax=500):
        b, a = signal.butter(8, (fmin, fmax), btype="bandpass", fs=fs)            #フィルタ伝達関数の分子と分母を計算
        y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
        return np.array(y)