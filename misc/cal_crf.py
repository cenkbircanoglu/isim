from functools import partial
from multiprocessing import Pool

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from misc.generate_label import generate_pseudo_label
from voc12 import my_collate


def calculate_crf(
    model,
    cfg,
    tr_loader,
    val_loader,
    dataset_train,
    dataset_valid,
    device,
    *args,
    **kwargs
):
    model.eval()
    tr_results = []
    val_results = []

    with torch.set_grad_enabled(False):
        print("\nCreating Pseudo Labels for training")
        for item in tqdm(tr_loader, total=len(tr_loader.dataset)):
            idx = item["idx"][0]
            img_i = [img_ii.to(device) for img_ii in item["img"]]
            res = generate_pseudo_label(model, img_i, item["label"][0], item["size"])
            tr_results.append((idx.cpu().item(), res))
        print("\nCreating Pseudo Labels for validation")
        for item in tqdm(val_loader, total=len(val_loader.dataset)):
            idx = item["idx"][0]
            img_i = [img_ii.to(device) for img_ii in item["img"]]
            res = generate_pseudo_label(model, img_i, item["label"][0], item["size"])
            val_results.append((idx.cpu().item(), res))
    print("Applying CRF to CAM results")
    if len(tr_results) > 0:
        with Pool(processes=4) as pool:
            pool.starmap(
                partial(dataset_train.update_cam, fg_thres=cfg.cam_eval_thres),
                tqdm(tr_results, total=len(tr_results)),
            )
    if len(val_results) > 0:
        with Pool(processes=4) as pool:
            pool.starmap(
                partial(dataset_valid.update_cam, fg_thres=cfg.cam_eval_thres),
                tqdm(val_results, total=len(val_results)),
            )

    loader_train = DataLoader(
        dataset_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=my_collate,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=cfg.batch_size,
        drop_last=False,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=my_collate,
    )
    return {"train": loader_train, "valid": loader_valid}
