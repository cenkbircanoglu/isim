import logging
import os

import hydra
import numpy as np
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion
from omegaconf import DictConfig
from tqdm import tqdm

from logger import Logger
from utils import log_loss_summary, set_seed

set_seed(3407)


@hydra.main(config_path="./conf/", config_name="eval_cam")
def run_app(cfg: DictConfig) -> None:
    logger = Logger(cfg.logs)
    try:
        epoch = cfg.cam_out_dir.split("/")[-2].split("-")[-1]
        print(epoch)
        epoch = int(epoch)
    except:
        epoch = cfg.last_epoch
    dataset = VOCSemanticSegmentationDataset(
        split=cfg.infer_set, data_dir=cfg.voc12_root
    )
    labels = [dataset.get_example_by_keys(i, (1,))[0] for i in range(len(dataset))]
    for subdir in os.listdir(cfg.cam_out_dir):
        if subdir.startswith("."):
            continue
        folder = os.path.join(cfg.cam_out_dir, subdir)
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

        results = dict(
            zip(
                [
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
                ],
                iou,
            )
        )
        results["miou"] = np.nanmean(iou)
        results["miou_without_background"] = np.nanmean(iou[1:])
        logging.info(f"iou: {iou}\tmiou: {np.nanmean(iou)}\tFolder: {folder}")
        for key, value in results.items():
            log_loss_summary(logger, value, epoch, tag=f"eval_{cfg.infer_set}_{key}")


if __name__ == "__main__":
    run_app()
