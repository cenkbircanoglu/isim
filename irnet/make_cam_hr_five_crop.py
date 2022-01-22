import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from misc import imutils
from models import initialize_model
from models.pipeline import ModelMode, ProcessMode
from voc12 import dataloader


def extract_valid_cams(outputs, size, label, cfg, img_name):
    strided_size = imutils.get_strided_size(size, 4)
    strided_up_size = imutils.get_strided_up_size(size, 16)

    strided_cams_list = [
        F.interpolate(
            torch.unsqueeze(o, 0), strided_size, mode="bilinear", align_corners=False
        )[0]
        for o in outputs
    ]
    strided_cams = torch.sum(torch.stack(strided_cams_list), dim=0)

    highres_cam = [
        F.interpolate(
            torch.unsqueeze(o, 1), strided_up_size, mode="bilinear", align_corners=False
        )
        for o in outputs
    ]
    highres_cam = torch.sum(torch.stack(highres_cam, 0), 0)[:, 0, : size[0], : size[1]]

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
        {"keys": keys, "cam": strided_cams.cpu(), "hr_cam": highres_cam.cpu().numpy()},
    )


@hydra.main(config_path="../conf/irnet/", config_name="make_cam_five_crop")
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
        if os.path.exists(os.path.join(cfg.output_dir, img_name + ".npy")):
            continue
        label = data["label"][0]
        imgs = data["img"]
        size = data["size"]
        label = label.to(device)

        with torch.set_grad_enabled(False):
            cams = []
            for img in imgs:
                max_side = max(img.shape[-2:]) // 2
                min_side = min(img.shape[-2:])
                five_size = 48
                if max_side > 48 and min_side > 96:
                    five_size = 96
                if max_side > 96 and min_side > 128:
                    five_size = 128
                if max_side > 128 and min_side > 256:
                    five_size = 256
                if max_side > 256 and min_side > 320:
                    five_size = 320
                if max_side > 320 and min_side > 448:
                    five_size = 448
                if max_side > 448 and min_side > 512:
                    five_size = 512
                if max_side > 512 and min_side > 720:
                    five_size = 720
                if max_side > 720 and min_side > 1024:
                    five_size = 1024

                five_crop = transforms.FiveCrop(size=(five_size, five_size))
                img_0 = five_crop(img[0][0])
                img_1 = five_crop(img[0][1])
                sub_img_size = img[0][0].shape[-2:]
                cam_zero = torch.zeros((20, sub_img_size[0], sub_img_size[1])).to(
                    device
                )
                five_cams = []
                for i in range(5):
                    img_i = torch.cat((img_0[i].unsqueeze(0), img_1[i].unsqueeze(0)), 0)
                    cam = model(
                        img_i.to(device),
                        model_mode=ModelMode.classification,
                        mode=ProcessMode.infer,
                    )
                    upsampled_cam = F.interpolate(
                        torch.unsqueeze(cam["cams"], 0),
                        (five_size, five_size),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                    five_cams.append(upsampled_cam)

                crop_top = int(round((int(sub_img_size[0]) - five_size) / 2.0))
                crop_left = int(round((int(sub_img_size[1]) - five_size) / 2.0))
                cam_zero[:, 0:five_size, 0:five_size] = torch.max(
                    cam_zero[:, 0:five_size, 0:five_size], five_cams[0]
                )
                cam_zero[:, 0:five_size, sub_img_size[1] - five_size :] = torch.max(
                    cam_zero[:, 0:five_size, sub_img_size[1] - five_size :],
                    five_cams[1],
                )
                cam_zero[:, sub_img_size[0] - five_size :, 0:five_size] = torch.max(
                    cam_zero[:, sub_img_size[0] - five_size :, 0:five_size],
                    five_cams[2],
                )
                cam_zero[
                    :, sub_img_size[0] - five_size :, sub_img_size[1] - five_size :
                ] = torch.max(
                    cam_zero[
                        :, sub_img_size[0] - five_size :, sub_img_size[1] - five_size :
                    ],
                    five_cams[3],
                )
                cam_zero[
                    :,
                    crop_top : crop_top + five_size,
                    crop_left : crop_left + five_size,
                ] = torch.max(
                    cam_zero[
                        :,
                        crop_top : crop_top + five_size,
                        crop_left : crop_left + five_size,
                    ],
                    five_cams[4],
                )

                cams.append(cam_zero)
            extract_valid_cams(cams, size, label, cfg, img_name)


def data_loaders(cfg):
    scales = [float(i) for i in str(cfg.scales).split("-")]
    print("Scales", scales)
    dataset = dataloader.VOC12ClassificationDatasetMSF(
        cfg.train_list, voc12_root=cfg.voc12_root, scales=scales
    )

    loader = DataLoader(
        dataset, drop_last=False, num_workers=1, persistent_workers=True
    )

    return loader


if __name__ == "__main__":
    run_app()
