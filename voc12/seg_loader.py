import os
from functools import partial

import hydra
import torch
from omegaconf import DictConfig
from torchvision import transforms

from voc12 import dataloader_2, custom_transforms, dataloader, my_collate
from voc12.dataloader import TorchvisionNormalize


def tmp_func(x, crop_size):
    return custom_transforms.random_resize_long_tuple(x, crop_size, crop_size * 2)


def data_loaders_old(cfg):
    os.makedirs(cfg.temp_dir, exist_ok=True)
    dataset_train = dataloader.VOC12PseudoSegmentationDataset(
        cfg.train_list,
        crop_size=cfg.crop_size,
        voc12_root=cfg.voc12_root,
        rescale=None,
        hor_flip=True,
        crop_method="random",
        resize_long=(cfg.crop_size, cfg.crop_size * 2),
        temp_dir=cfg.temp_dir,
    )

    dataset_valid = dataloader.VOC12PseudoSegmentationDataset(
        cfg.val_list,
        crop_size=cfg.crop_size,
        img_normal=TorchvisionNormalize(),
        voc12_root=cfg.voc12_root,
        temp_dir=cfg.temp_dir,
    )
    scales = [float(i) for i in str(cfg.scales).split("-")]
    print("Scales", scales)
    train_dataset = dataloader.VOC12ClassificationDatasetMSF(
        cfg.train_list, voc12_root=cfg.voc12_root, scales=scales
    )
    valid_dataset = dataloader.VOC12ClassificationDatasetMSF(
        cfg.val_list, voc12_root=cfg.voc12_root, scales=scales
    )

    return dataset_train, dataset_valid, train_dataset, valid_dataset


def data_loaders(cfg):
    crop_size = cfg.crop_size
    tr_transform = transforms.Compose(
        [
            custom_transforms.CustomPILToTensor(),
            # custom_transforms.CustomRandomAdjustSharpness(sharpness_factor=2, p=0.1),
            # custom_transforms.CustomRandomAutocontrast(p=0.1),
            # custom_transforms.CustomRandomEqualize(p=0.1),
            custom_transforms.CustomRandomRotation(degrees=(0, 10)),
            custom_transforms.CustomRandomHorizontalFlip(0.5),
            transforms.Lambda(partial(tmp_func, crop_size=crop_size)),
            custom_transforms.CustomRandomCrop(
                size=(crop_size, crop_size), pad_if_needed=True
            ),
            custom_transforms.CustomConvertImageDtype(torch.float),
            custom_transforms.CustomRandomErasing(p=0.01, scale=(0.02, 0.2)),
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
    dataset_train = dataloader_2.VOC12PseudoSegmentationDataset(
        cfg.train_list,
        voc12_root=cfg.voc12_root,
        transform=tr_transform,
        temp_dir=cfg.temp_dir,
    )

    dataset_valid = dataloader_2.VOC12PseudoSegmentationDataset(
        cfg.val_list,
        transform=test_transform,
        voc12_root=cfg.voc12_root,
        temp_dir=cfg.temp_dir,
    )
    scales = [float(i) for i in str(cfg.scales).split("-")]
    print("Scales", scales)
    transform = transforms.Compose(
        [
            custom_transforms.CustomPILToTensor(),
            custom_transforms.CustomConvertImageDtype(torch.float),
            custom_transforms.CustomNormalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    train_dataset = dataloader_2.VOC12ClassificationDatasetMSF(
        cfg.train_list, voc12_root=cfg.voc12_root, scales=scales, transform=transform
    )
    valid_dataset = dataloader_2.VOC12ClassificationDatasetMSF(
        cfg.val_list, voc12_root=cfg.voc12_root, scales=scales, transform=transform
    )

    return dataset_train, dataset_valid, train_dataset, valid_dataset


@hydra.main(config_path="../conf/", config_name="train_d")
def run_app(cfg: DictConfig) -> None:
    import time

    a, b, c, d = data_loaders_old(cfg)
    from torch.utils.data import DataLoader

    loader_train = DataLoader(
        a,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=my_collate,
    )
    start = time.time()
    for i in loader_train:
        pass
    stop = time.time()

    print(stop - start)

    import time

    a, b, c, d = data_loaders(cfg)
    loader_train = DataLoader(
        a,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=my_collate,
    )
    start = time.time()
    for i in loader_train:
        pass
    stop = time.time()

    print(stop - start)


if __name__ == "__main__":
    run_app()
