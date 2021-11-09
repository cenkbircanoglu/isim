import numpy as np
from torch.utils.data import DataLoader

from voc12 import dataloader


def data_loaders(cfg):
    dataset_train = dataloader.VOC12ClassificationDataset(
        cfg.train_list,
        voc12_root=cfg.voc12_root,
        # resize_long=(cfg.crop_size // 2, cfg.crop_size * 2),
        resize_long=(cfg.crop_size, cfg.crop_size * 2),
        hor_flip=True,
        crop_size=cfg.crop_size,
        crop_method="random",
    )
    dataset_valid = dataloader.VOC12ClassificationDataset(
        cfg.val_list, voc12_root=cfg.voc12_root, crop_size=cfg.crop_size
    )

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.workers,
        worker_init_fn=worker_init,
        persistent_workers=True,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=cfg.batch_size,
        drop_last=False,
        num_workers=cfg.workers,
        worker_init_fn=worker_init,
        persistent_workers=True,
    )

    return loader_train, loader_valid
