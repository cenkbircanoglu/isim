import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_loaders import my_collate
from models import initialize_model
from models.pipeline import ModelMode, ProcessMode
from voc12 import dataloader


def extract_valid_cams(cams, size, label, cfg, img_name):
    output_folder = os.path.join(cfg.output_dir, "0")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, img_name + ".npy")
    if not os.path.exists(output_path):
        strided_cam = [
            F.interpolate(
                torch.unsqueeze(o, 1), size, mode="bilinear", align_corners=False
            )
            for o in cams
        ]
        strided_cam = torch.sum(torch.stack(strided_cam, 0), 0)[
            :, 0, : size[0], : size[1]
        ]

        valid_cat = torch.nonzero(label, as_tuple=False)[:, 0]

        strided_cam = strided_cam[valid_cat]
        strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

        np.save(
            output_path,
            {"keys": valid_cat.cpu().numpy(), "high_res": strided_cam.cpu().numpy()},
        )


@hydra.main(config_path="./conf/", config_name="make_cam")
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
    dataset = dataloader.VOC12ClassificationDatasetMSF(
        cfg.infer_list, voc12_root=cfg.voc12_root, scales=[1.0, 0.5, 1.5, 2.0]
    )

    loader = DataLoader(
        dataset,
        drop_last=False,
        num_workers=0,
        persistent_workers=False,
        collate_fn=my_collate,
    )

    return loader


if __name__ == "__main__":
    run_app()
