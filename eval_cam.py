import logging
import os

import hydra
import numpy as np
import wandb
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from omegaconf import DictConfig
from tqdm import tqdm

from logger import Logger
from utils import log_loss_summary, set_seed

set_seed(3407)
CATS = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


@hydra.main(config_path="./conf/", config_name="eval_cam")
def run_app(cfg: DictConfig) -> None:
    run = wandb.init(
        project=f"{cfg.wandb.project}",
        name=cfg.wandb.name,
        config=cfg.__dict__,
        tags=["eval", "pipeline"],
    )
    logger = Logger(cfg.logs)
    epochs = cfg.epochs.split(",")
    dataset = VOCSemanticSegmentationDataset(
        split=cfg.infer_set, data_dir=cfg.voc12_root
    )
    labels = [
        dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))
    ]
    for iter, epoch in enumerate(epochs):
        cam_out_dir = f"{cfg.cam_out_dir}{epoch}/cam_outputs"
        for subdir in os.listdir(cam_out_dir):
            if subdir.startswith("."):
                continue
            folder = os.path.join(cam_out_dir, subdir)
            preds = []
            for id in tqdm(dataset.ids):
                cam_dict = np.load(
                    os.path.join(folder, id + ".npy"), allow_pickle=True
                ).item()
                cams = cam_dict["high_res"]
                cams = np.pad(
                    cams,
                    ((1, 0), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=cfg.cam_eval_thres,
                )
                keys = np.pad(cam_dict["keys"] + 1, (1, 0), mode="constant")
                cls_labels = np.argmax(cams, axis=0)
                cls_labels = keys[cls_labels]
                preds.append(cls_labels.copy())

            confusion = calc_semantic_segmentation_confusion(preds, labels)

            gtj = confusion.sum(axis=1)
            resj = confusion.sum(axis=0)
            gtjresj = np.diag(confusion)
            denominator = gtj + resj - gtjresj
            iou = gtjresj / denominator

            results = dict(zip(CATS, iou))
            results["miou"] = np.nanmean(iou)
            results["miou_without_background"] = np.nanmean(iou[1:])
            logging.info(
                f"iou: {iou}\tmiou: {np.nanmean(iou)}\tFolder: {folder}"
            )
            for key, value in results.items():
                log_loss_summary(
                    logger, value, iter, tag=f"eval_{cfg.infer_set}_{key}"
                )
    run.finish()


if __name__ == "__main__":
    run_app()
