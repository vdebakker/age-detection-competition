import numpy as np
from numba import jit
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

import aug_lib

classes = ['YOUNG', 'MIDDLE', 'OLD']

train_data_path = 'data/train/'
test_data_path = 'data/test/'

means = torch.tensor([0.485, 0.456, 0.406])
stds = torch.tensor([0.229, 0.224, 0.225])


def encode_class(df, smoothing=0):
    class2vec = {
        'YOUNG': np.array([1 - smoothing, smoothing, 0]),
        'MIDDLE': np.array([smoothing, 1 - 2 * smoothing, smoothing]),
        'OLD': np.array([0, smoothing, 1 - smoothing])}
    df['Class'] = df['Class'].transform(lambda x: class2vec[x])
    return df


def read_img(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


@jit(nopython=True)
def pad(img):
    h, w, _ = img.shape
    max_wh = max(w, h)
    h_padding = (max_wh - w) // 2
    v_padding = (max_wh - h) // 2

    new_img = np.zeros((max_wh, max_wh, 3), dtype=np.uint8)
    new_img[v_padding: v_padding + h, h_padding: h_padding + w] = img

    return new_img


class AgeDataset(Dataset):
    def __init__(self, df, transforms, include_labels=True, augment=False):
        self.imgs = df['ID'].to_numpy()
        if include_labels:
            self.ages = torch.tensor(np.array(list(df['Class'].to_numpy())))
            assert len(self.ages) == len(self.imgs)

        self.include_labels = include_labels
        self.transforms = transforms
        self.augment = augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = read_img(self.imgs[idx])
        #img = pad(img)

        #if self.augment:
        #    img = np.array(augmentations(img))
        img = self.transforms(image=img)["image"]

        if not self.include_labels:
            return img
        age = self.ages[idx]
        return img, age


augmenter = aug_lib.TrivialAugment()
augmentations = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                torchvision.transforms.Lambda(augmenter)])


def get_augment_transforms(p_augment: float, p_augment2: float):
   return A.Compose([
       #A.LongestMaxSize(224),
       #A.RandomResizedCrop(height=224, width=224, scale=(.6, 1.)),
       A.Resize(height=224, width=224),
       A.HorizontalFlip(p=0.5),
       A.GaussNoise(p=p_augment2),
       A.OneOf([
           A.MotionBlur(p=0.2),
           A.MedianBlur(blur_limit=11, p=0.2),
           A.Blur(blur_limit=11, p=0.2),
           A.Downscale(p=.2),
       ], p=p_augment2),
       A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                          rotate_limit=45, p=p_augment2),
       A.OneOf([
           A.OpticalDistortion(p=0.3),
           A.GridDistortion(p=0.1),
           A.PiecewiseAffine(p=0.3),
       ], p=p_augment2),
       A.OneOf([
           A.Sharpen(),
           A.Emboss(),
           A.RandomContrast(),
           A.RandomBrightness(),
           A.CLAHE(),
           A.HueSaturationValue(),
       ], p=p_augment2),
       A.ToGray(p=.1)
   ], p=p_augment)


def get_dataloader(df, batch_size=32, include_labels=True, augment_prob=.0, augment_prob2=0., smoothing=0, num_workers=8):
    transforms = A.Compose([
        #A.LongestMaxSize(224),
        #A.PadIfNeeded(224, 224, border_mode=cv2.BORDER_REFLECT_101),
        A.Resize(height=224, width=224),
        A.Normalize(mean=means, std=stds),
        ToTensorV2()
    ])

    if include_labels:
        df = df.copy()
        encode_class(df, smoothing)

    transforms = A.Compose([get_augment_transforms(
        p_augment=augment_prob, p_augment2=augment_prob2), transforms])

    ds = AgeDataset(df=df, transforms=transforms,
                    include_labels=include_labels)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
