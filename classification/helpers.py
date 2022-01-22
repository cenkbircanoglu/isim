import torch
from tqdm import tqdm

from losses import mlsm_loss
from models.pipeline import ModelMode, ProcessMode
from utils import get_ap_score, set_seed

set_seed(9)
torch.backends.cudnn.benchmark = True


def train_pipeline_one_epoch(model, dataset_loader, optimizer, scheduler=None):
    model.train()
    total_cnt = total_cls_loss = total_ap_score = 0.0
    for i, batch in tqdm(
        enumerate(dataset_loader),
        total=len(dataset_loader.dataset) // dataset_loader.batch_size,
    ):
        optimizer.zero_grad(set_to_none=True)
        img, cls_label = batch["img"].cuda(), batch["label"].cuda()

        batch_size = cls_label.size(0)
        total_cnt += batch_size

        with torch.set_grad_enabled(True):
            cls_logits = model(
                img, model_mode=ModelMode.classification, mode=ProcessMode.train
            )["cls"]
            loss = mlsm_loss(cls_logits, cls_label)
            total_cls_loss += loss.item()
        with torch.set_grad_enabled(False):
            total_ap_score += get_ap_score(
                cls_label.cpu().detach().numpy(),
                torch.sigmoid(cls_logits).cpu().detach().numpy(),
            )

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    avg_cls_acc, avg_cls_loss = total_ap_score / total_cnt, total_cls_loss / total_cnt

    return model, avg_cls_acc, avg_cls_loss


def eval_pipeline_one_epoch(model, dataset_loader):
    model.eval()
    total_cnt = total_cls_loss = total_ap_score = 0.0
    for i, batch in tqdm(
        enumerate(dataset_loader),
        total=len(dataset_loader.dataset) // dataset_loader.batch_size,
    ):
        with torch.no_grad():
            img, cls_label = batch["img"].cuda(), batch["label"].cuda()
            batch_size = cls_label.size(0)
            total_cnt += batch_size

            cls_logits = model(
                img, model_mode=ModelMode.classification, mode=ProcessMode.train
            )["cls"]
            cls_loss = mlsm_loss(cls_logits, cls_label)
            total_cls_loss = cls_loss.item()
            total_ap_score += get_ap_score(
                cls_label.cpu().detach().numpy(),
                torch.sigmoid(cls_logits).cpu().detach().numpy(),
            )

    avg_cls_acc, avg_cls_loss = total_ap_score / total_cnt, total_cls_loss / total_cnt

    return avg_cls_acc, avg_cls_loss
