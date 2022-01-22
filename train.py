import logging
import os
from functools import partial
from uuid import uuid4

import hydra
import numpy as np
import torch
import wandb
from PIL import Image
from chainercv.evaluations import calc_semantic_segmentation_confusion
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_loaders import seg_loader, my_collate
from logger import Logger
from losses import modified_cross_entropy_loss, mlsm_loss
from misc import torchutils
from misc.cal_crf import calculate_crf
from models import initialize_model
from models.pipeline import ModelMode, ProcessMode
from utils import get_ap_score, makedirs, log_images, log_loss_summary, set_seed


def train_pipeline_one_epoch(model, dataset_loader, optimizer, epoch, scaler=None):
    model.train()
    total_cnt = total_cls_loss = total_seg_loss = total_ap_score = 0.0
    for i, batch in tqdm(
            enumerate(dataset_loader),
            total=len(dataset_loader.dataset) // dataset_loader.batch_size,
    ):
        optimizer.zero_grad(set_to_none=True)
        img, cls_label, seg_label = (
            batch["img"].cuda(),
            batch["label"].cuda(),
            batch["seg_label"].long().cuda(),
        )
        batch_size = cls_label.size(0)
        total_cnt += batch_size
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            with torch.set_grad_enabled(True):
                d = model(
                    img, model_mode=ModelMode.segmentation, mode=ProcessMode.train
                )
                cls_logits, seg_logits = d["cls"], d["seg"]
                cls_loss = mlsm_loss(cls_logits, cls_label)
                seg_loss = modified_cross_entropy_loss(seg_logits, seg_label)
                total_cls_loss += cls_loss.item()
                total_seg_loss += seg_loss.item()
                loss = cls_loss + seg_loss
        with torch.set_grad_enabled(False):
            total_ap_score += get_ap_score(
                cls_label.cpu().detach().numpy(),
                torch.sigmoid(cls_logits).cpu().detach().numpy(),
            )
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer, epoch=epoch)
            scaler.update()
        else:
            loss.backward()
            optimizer.step(epoch=epoch)
    avg_cls_acc, avg_cls_loss, avg_seg_loss = (
        total_ap_score / total_cnt,
        total_cls_loss / total_cnt,
        total_seg_loss / total_cnt,
    )
    return model, avg_cls_acc, avg_cls_loss, avg_seg_loss


def eval_pipeline_one_epoch(model, dataset_loader, epoch, logger, cfg):
    model.eval()
    total_cnt = total_cls_loss = total_seg_loss = total_ap_score = 0.0
    for i, batch in tqdm(
            enumerate(dataset_loader),
            total=len(dataset_loader.dataset) // dataset_loader.batch_size,
    ):
        with torch.no_grad():
            img, cls_label, seg_label = (
                batch["img"].cuda(),
                batch["label"].cuda(),
                batch["seg_label"].long().cuda(),
            )
            batch_size = cls_label.size(0)
            total_cnt += batch_size

            d = model(
                img, model_mode=ModelMode.segmentation, mode=ProcessMode.train
            )
            cls_logits, seg_logits = d["cls"], d["seg"]
            cls_loss = mlsm_loss(cls_logits, cls_label)
            seg_loss = modified_cross_entropy_loss(seg_logits, seg_label)
            total_cls_loss += cls_loss.item()
            total_seg_loss += seg_loss.item()
            total_ap_score += get_ap_score(
                cls_label.cpu().detach().numpy(),
                torch.sigmoid(cls_logits).cpu().detach().numpy(),
            )
            if i * cfg.batch_size < cfg.vis_images:
                tag = "image/{}".format(i)
                num_images = cfg.vis_images - i * cfg.batch_size
                logger.image_list_summary(
                    tag,
                    log_images(img, seg_label, seg_logits)[:num_images],
                    epoch,
                )
    avg_cls_acc, avg_cls_loss, avg_seg_loss = (
        total_ap_score / total_cnt,
        total_cls_loss / total_cnt,
        total_seg_loss / total_cnt,
    )
    return avg_cls_acc, avg_cls_loss, avg_seg_loss


def calculate_segmentation_metric(dataset, epoch, data_type="train"):
    preds = []
    labels = []

    for item in tqdm(dataset):
        seg_label, org_seg_label = (item["seg_label"], item["original_label"])

        seg_label_resized = Image.fromarray(seg_label).resize(
            org_seg_label.shape[::-1], resample=Image.BILINEAR
        )
        seg_label_resized = np.array(seg_label_resized).astype(np.int32)
        preds.append(seg_label_resized.copy())
        labels.append(org_seg_label.copy())

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
    print(results, f"miou: {np.nanmean(iou)}", data_type)
    wandb.log(results, step=epoch)
    return float(np.nanmean(iou))


@hydra.main(config_path="./conf/", config_name="train")
def run_app(cfg: DictConfig) -> None:
    run = wandb.init(
        project=f"{cfg.wandb.project}",
        name=cfg.wandb.name,
        config=cfg.__dict__,
        tags=["train", "pipeline"],
    )
    makedirs(cfg)
    (
        dataset_train,
        dataset_valid,
        tr_data_scaled,
        val_data_scaled,
    ) = seg_loader.data_loaders(cfg)
    logger = Logger(cfg.logs)
    model = initialize_model(cfg)
    param_groups = model.trainable_parameters()
    model = torch.nn.DataParallel(model).cuda()
    wandb.watch(model)
    tr_loader = DataLoader(
        tr_data_scaled,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=my_collate,
    )
    val_loader = DataLoader(
        val_data_scaled,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=my_collate,
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

    max_step = (len(dataset_train) // cfg.batch_size) * cfg.max_step

    optimizer = torchutils.PolyOptimizer(
        [
            {
                "params": param_groups[0],
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 10 * cfg.lr,
                "weight_decay": cfg.weight_decay,
            },
            {
                "params": param_groups[2],
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
            },
        ],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        max_step=max_step,
        logger=logger,
    )
    scaler = torch.cuda.amp.GradScaler()
    crf_counter = 0
    for epoch in tqdm(range(cfg.epochs), total=cfg.epochs):

        if epoch == 0:
            miou = calculate_segmentation_metric(loader_valid.dataset, epoch, data_type="train")
            log_loss_summary(logger, miou, epoch, tag=f"val_miou")

        model, tr_cls_acc, tr_cls_loss, tr_seg_loss = train_pipeline_one_epoch(
            model, loader_train, optimizer, epoch, scaler
        )
        logging.info(
            f"\nEpoch: {epoch}\tData: Train\tAverage Cls Acc: {tr_cls_acc}\tAverage Cls Loss: {tr_cls_loss}\tAverage Seg Loss {tr_seg_loss}\n"
        )
        log_loss_summary(logger, float(tr_cls_acc), epoch, tag=f"tr_cls_acc")
        log_loss_summary(logger, float(tr_cls_loss), epoch, tag=f"tr_cls_loss")
        log_loss_summary(logger, float(tr_seg_loss), epoch, tag=f"tr_seg_loss")
        val_cls_acc, val_cls_loss, val_seg_loss = eval_pipeline_one_epoch(
            model, loader_valid, epoch, logger, cfg
        )
        logging.info(
            f"\nEpoch: {epoch}\tData: Val\tAverage Cls Acc: {val_cls_acc}\tAverage Cls Loss: {val_cls_loss}\tAverage Seg Loss {val_seg_loss}\n"
        )
        log_loss_summary(logger, float(val_cls_acc), epoch, tag=f"val_cls_acc")
        log_loss_summary(logger, float(val_cls_loss), epoch, tag=f"val_cls_loss")
        log_loss_summary(logger, float(val_seg_loss), epoch, tag=f"val_seg_loss")
        if ((epoch + 1) % cfg.crf_freq == 0 and epoch != cfg.epochs - 1) or (
                epoch + 1
        ) == 5:
            if crf_counter == cfg.crf_counter:
                logging.info(f'Stopping the training as it reached the crf_counter: {cfg.crf_counter}, {crf_counter}')
                break
            torch.save(
                model.module.state_dict(),
                os.path.join(cfg.weights, f"seg-model-{epoch}.pth"),
            )
            logging.info(
                f"Regenerating the segmentation labels! crf counter: {crf_counter} and freq: {cfg.crf_freq}"
            )
            loaders = calculate_crf(
                model.module,
                cfg,
                tr_loader,
                val_loader,
                dataset_train,
                dataset_valid,
                "cuda",
            )
            loader_train = loaders["train"]
            loader_valid = loaders["valid"]
            miou = calculate_segmentation_metric(loader_valid.dataset, epoch, data_type="train")
            log_loss_summary(logger, miou, epoch, tag=f"train_miou")
            crf_counter += 1
            log_loss_summary(logger, crf_counter, epoch, tag=f"crf_counter")
        torch.save(
            model.module.state_dict(),
            os.path.join(cfg.weights, "final-model.pth"),
        )
    logging.info("Training Finished")
    artifact = wandb.Artifact(str(uuid4()), type="model")
    artifact.add_file(os.path.join(cfg.weights, "final-model.pth"))
    run.log_artifact(artifact)
    logging.info("Artifacts Saved")
    run.finish()
    logging.info("Run Finished")


if __name__ == "__main__":
    set_seed(9)
    torch.backends.cudnn.benchmark = True
    torch.autograd.profiler.profile(False)
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    run_app()
