import os

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from classification.helpers import train_pipeline_one_epoch, eval_pipeline_one_epoch
from logger import Logger
from models import initialize_model
from utils import makedirs, log_loss_summary, set_seed
from voc12 import cls_loader, my_collate

set_seed(9)
torch.backends.cudnn.benchmark = True


@hydra.main(config_path="../conf/classification", config_name="train")
def run_app(cfg: DictConfig) -> None:
    makedirs(cfg)
    dataset_train, dataset_valid = cls_loader.data_loaders(cfg)
    logger = Logger(cfg.logs)
    model = initialize_model(cfg)
    model = model.cuda()

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

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    steps_per_epoch = len(dataset_train) // cfg.batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, steps_per_epoch=steps_per_epoch, epochs=cfg.epochs
    )
    best_val_loss = float("inf")
    patience_cnt = 0
    for epoch in tqdm(range(cfg.epochs), total=cfg.epochs):
        model, tr_cls_acc, tr_cls_loss = train_pipeline_one_epoch(
            model, loader_train, optimizer, scheduler
        )
        print(
            f"\nEpoch: {epoch}\tData: Train\tAverage Cls Acc: {tr_cls_acc}\tAverage Cls Loss: {tr_cls_loss}\n"
        )
        log_loss_summary(logger, float(tr_cls_acc), epoch, tag=f"tr_cls_acc")
        log_loss_summary(logger, float(tr_cls_loss), epoch, tag=f"tr_cls_loss")
        val_cls_acc, val_cls_loss = eval_pipeline_one_epoch(model, loader_valid)
        print(
            f"\nEpoch: {epoch}\tData: Val\tAverage Cls Acc: {val_cls_acc}\tAverage Cls Loss: {val_cls_loss}\n"
        )
        log_loss_summary(logger, float(val_cls_acc), epoch, tag=f"val_cls_acc")
        log_loss_summary(logger, float(val_cls_loss), epoch, tag=f"val_cls_loss")
        if val_cls_loss < best_val_loss:
            print(f"Saving best model in epoch {epoch} with loss {val_cls_loss}")
            best_val_loss = val_cls_loss
            patience_cnt = 0
            torch.save(
                model.state_dict(),
                os.path.join(cfg.weights, f"best-model.pt"),
            )
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                break
    print("Training Finished")


if __name__ == "__main__":
    run_app()
