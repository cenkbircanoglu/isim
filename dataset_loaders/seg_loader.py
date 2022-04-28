import logging
import os

from voc12 import dataloader
from voc12.dataloader import TorchvisionNormalize


def data_loaders(cfg):
    os.makedirs(cfg.temp_dir, exist_ok=True)
    dataset_train = dataloader.VOC12PseudoSegmentationDataset(
        cfg.train_list,
        crop_size=cfg.crop_size,
        voc12_root=cfg.voc12_root,
        rescale=None,
        hor_flip=True,
        crop_method="random",
        # resize_long=(cfg.crop_size // 2, cfg.crop_size * 2),
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
    logging.info(f"Scales {str(scales)}")
    train_dataset = dataloader.VOC12ClassificationDatasetMSF(
        cfg.train_list, voc12_root=cfg.voc12_root, scales=scales
    )
    valid_dataset = dataloader.VOC12ClassificationDatasetMSF(
        cfg.val_list, voc12_root=cfg.voc12_root, scales=scales
    )

    return dataset_train, dataset_valid, train_dataset, valid_dataset
