import logging
import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from misc import imutils
from models import initialize_model
from models.pipeline import ModelMode, ProcessMode
from utils import set_seed
from voc12 import dataloader

set_seed(3407)


def extract_valid_cams(outputs, size, label, cfg, img_name):
    strided_size = imutils.get_strided_size(size, 4)
    strided_up_size = imutils.get_strided_up_size(size, 16)

    strided_cams_list = [
        F.interpolate(
            torch.unsqueeze(o, 0),
            strided_size,
            mode="bilinear",
            align_corners=False,
        )[0]
        for o in outputs
    ]
    strided_cams = torch.sum(torch.stack(strided_cams_list), dim=0)

    highres_cam = [
        F.interpolate(
            torch.unsqueeze(o, 1),
            strided_up_size,
            mode="bilinear",
            align_corners=False,
        )
        for o in outputs
    ]
    highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[
        :, 0, : size[0], : size[1]
    ]

    keys = torch.nonzero(label, as_tuple=False)[:, 0]

    strided_cams = strided_cams[keys]
    strided_cams /= F.adaptive_max_pool2d(strided_cams, (1, 1)) + 1e-5

    highres_cam = highres_cam[keys]
    highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5
    output_folder = cfg.output_dir
    os.makedirs(output_folder, exist_ok=True)
    keys = np.pad(keys.cpu() + 1, (1, 0), mode="constant")
    np.save(
        os.path.join(output_folder, img_name + ".npy"),
        {
            "keys": keys,
            "cam": strided_cams.cpu(),
            "hr_cam": highres_cam.cpu().numpy(),
        },
    )


@hydra.main(config_path="../conf/irnet/", config_name="make_cam")
def run_app(cfg: DictConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cpu" if not torch.cuda.is_available() else cfg.device)

    data_loader = data_loaders(cfg)
    model = initialize_model(cfg)
    model.load_state_dict(torch.load(cfg.weights), strict=True)
    model.to(device)
    model.eval()

    for i, data in tqdm(enumerate(data_loader), total=len(data_loader.dataset)):
        img_name = data["name"][0]
        label = data["label"][0]
        imgs = data["img"]
        size = data["size"]
        label = label.to(device)

        with torch.set_grad_enabled(False):
            cams = []
            for img in imgs:
                cam = model(
                    img[0].to(device),
                    model_mode=ModelMode.classification,
                    mode=ProcessMode.infer,
                )
                cams.append(cam["cams"])
            extract_valid_cams(cams, size, label, cfg, img_name)


def data_loaders(cfg):
    scales = [float(i) for i in str(cfg.scales).split("-")]
    logging.info(f"Scales {str(scales)}")
    dataset = dataloader.VOC12ClassificationDatasetMSF(
        cfg.train_list, voc12_root=cfg.voc12_root, scales=scales
    )

    loader = DataLoader(
        dataset, drop_last=False, num_workers=1, persistent_workers=True
    )

    return loader


if __name__ == "__main__":
    run_app()
