import os
from functools import partial

import torch
from torchvision import transforms

from voc12 import dataloader_2, custom_transforms


def tmp_func(x, crop_size):
    return custom_transforms.random_resize_long_tuple(x, crop_size, crop_size * 2)


def data_loaders(cfg):
    crop_size = cfg.crop_size
    tr_transform = transforms.Compose(
        [
            custom_transforms.CustomPILToTensor(),
            custom_transforms.CustomRandomRotation(degrees=(0, 10)),
            custom_transforms.CustomRandomHorizontalFlip(0.5),
            transforms.Lambda(partial(tmp_func, crop_size=crop_size)),
            custom_transforms.CustomRandomCrop(
                size=(crop_size, crop_size), pad_if_needed=True
            ),
            custom_transforms.CustomConvertImageDtype(torch.float),
            custom_transforms.CustomRandomErasing(p=0.1, scale=(0.02, 0.2)),
            custom_transforms.CustomNormalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            custom_transforms.CustomPILToTensor(),
            custom_transforms.CustomCenterCrop(size=crop_size),
            custom_transforms.CustomConvertImageDtype(torch.float),
            custom_transforms.CustomNormalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    os.makedirs(cfg.temp_dir, exist_ok=True)
    dataset_train = dataloader_2.VOC12ClassificationDataset(
        cfg.train_list, voc12_root=cfg.voc12_root, transform=tr_transform
    )

    dataset_valid = dataloader_2.VOC12ClassificationDataset(
        cfg.val_list, transform=test_transform, voc12_root=cfg.voc12_root
    )

    return dataset_train, dataset_valid
