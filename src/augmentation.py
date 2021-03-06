from albumentations import (
    PadIfNeeded, HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize,ToGray
)

from albumentations.pytorch import ToTensorV2

"""
def get_train_transforms(image_size):
    return Compose([
        Resize(image_size, image_size),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
        HueSaturationValue(p=0.5, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
        ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
        CoarseDropout(p=0.5),
        Normalize()
    ])

def get_inference_transforms(image_size):
    return Compose([
        Resize(image_size, image_size),
        Normalize()
    ])
"""
def get_train_transforms(config):
    trans_list = []
    if config["augmentation"]["resize"]:
        if config["learn_resize"]:
            trans_list.append(Resize(config["img_size"] * 2, config["img_size"] * 2))
        else:
            trans_list.append(Resize(config["img_size"], config["img_size"]))
    if config["augmentation"]["H_flip"]:
        trans_list.append(HorizontalFlip(p=0.5))
    if config["augmentation"]["V_flip"]:
        trans_list.append(VerticalFlip(p=0.5))
    if config["augmentation"]["Normalize"]:
        trans_list.append(
            Normalize(mean=[0.485]*config["in_channels"], std=[0.229]*config["in_channels"])
        )
    trans_list.append(ToTensorV2(p=1.0))

    return Compose(trans_list, p=1.)
    
    if config["augmentation"]["resize"]:
        return Compose([
            Resize(config["img_size"], config["img_size"]),
            # HorizontalFlip(p=0.5),
            # VerticalFlip(p=0.5),
            # ShiftScaleRotate(p=0.5),
            # MotionBlur(p=.2),
            # IAASharpen(p=.25),
            # Normalize(
            #     mean=[0.485],
            #     std=[0.229],
            # ),
            ToTensorV2(p=1.0),
        ], p=1.)
    else:
        return Compose([
            # Resize(config["img_size"], config["img_size"]),
            # HorizontalFlip(p=0.5), 
            # MotionBlur(p=.2),
            # IAASharpen(p=.25),
            # Normalize(
            #     mean=[0.485],
            #     std=[0.229],
            # ),
            ToTensorV2(p=1.0),
        ], p=1.)

def get_valid_transforms(config):
    trans_list = []
    if config["augmentation"]["resize"]:
        trans_list.append(Resize(config["img_size"], config["img_size"]))
    if config["augmentation"]["Normalize"]:
        trans_list.append(
            Normalize(mean=[0.485]*config["in_channels"], std=[0.229]*config["in_channels"])
        )
    trans_list.append(ToTensorV2(p=1.0))

    return Compose(trans_list, p=1.)

def get_inference_transforms(config):
    trans_list = []
    if config["augmentation"]["resize"]:
        if config["learn_resize"]:
            trans_list.append(Resize(config["img_size"] * 2, config["img_size"] * 2))
        else:
            trans_list.append(Resize(config["img_size"], config["img_size"]))
    if config["augmentation"]["H_flip"]:
        trans_list.append(HorizontalFlip(p=0.5))
    if config["augmentation"]["V_flip"]:
        trans_list.append(VerticalFlip(p=0.5))
    if config["augmentation"]["Normalize"]:
        trans_list.append(
            Normalize(mean=[0.485]*config["in_channels"], std=[0.229]*config["in_channels"])
        )
    trans_list.append(ToTensorV2(p=1.0))

    return Compose(trans_list, p=1.)
