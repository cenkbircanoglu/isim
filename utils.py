import os
import random

import numpy as np
import torch
import wandb
from skimage.color import label2rgb
from sklearn.metrics import average_precision_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_loss_summary(logger, loss, step, tag):
    wandb.log({tag: np.mean(loss)}, step=step)
    logger.scalar_summary(tag, np.mean(loss), step)


def makedirs(cfg):
    os.makedirs(cfg.weights, exist_ok=True)
    os.makedirs(cfg.logs, exist_ok=True)


def get_ap_score(y_true, y_scores):
    scores = 0.0

    for i in range(y_true.shape[0]):
        scores += average_precision_score(y_true=y_true[i], y_score=y_scores[i])

    return scores


def log_images(x, y_true, y_pred):
    images = []
    x_np = x.cpu().numpy()
    x_np = np.transpose(x_np, (0, 2, 3, 1))
    if type(y_pred) == tuple:
        y_pred = y_pred[0]
    for i in range(x_np.shape[0]):
        y_pred_np = np.argmax(y_pred[i].cpu().detach().numpy(), axis=0)

        image = x_np[i]
        image += np.abs(np.min(image))
        image_max = np.abs(np.max(image))
        if image_max > 0:
            image /= image_max
        image *= 255
        prediction_segmentation = (
            label2rgb(y_pred_np.astype(np.uint8), bg_label=0) * 255
        )
        true_segmentation = label2rgb(y_true[i].cpu().numpy(), bg_label=0) * 255
        images.append(image)
        images.append(true_segmentation)
        images.append(prediction_segmentation)
    return images
