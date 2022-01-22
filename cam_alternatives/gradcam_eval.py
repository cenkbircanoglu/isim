import os

import hydra
import numpy as np
import torch
from cv2 import cv2
from omegaconf import DictConfig
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm

from cam_alternatives.pipeline import initialize_model
from make_cam import data_loaders
from voc12.dataloader import get_img_path


@hydra.main(config_path="../conf/", config_name="gradcam")
def run_app(cfg: DictConfig) -> None:
    model = initialize_model(cfg)
    target_layers = [model.encoder_model.stage5[-1]]
    os.makedirs(cfg.output_dir, exist_ok=True)
    data_loader = data_loaders(cfg)
    model.load_state_dict(torch.load(cfg.weights), strict=False)
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader.dataset)):
        img_name = data["name"][0]
        image_path = get_img_path(img_name, cfg.voc12_root)
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255

        label = data["label"][0]
        imgs = data["img"]
        image = imgs[0][0][0]
        output_folder = os.path.join(cfg.output_dir, "0")
        output_path = os.path.join(output_folder, img_name + ".npy")
        if not os.path.exists(output_path):

            # Construct the CAM object once, and then re-use it on many images:
            cam = GradCAMPlusPlus(
                model=model, target_layers=target_layers, use_cuda=True
            )

            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            for target_category in torch.where(label == 1)[0].tolist():
                grayscale_cam = cam(
                    input_tensor=torch.unsqueeze(image, dim=0),
                    target_category=target_category,
                )

                # In this example grayscale_cam has only one image in the batch:
                grayscale_cam = grayscale_cam[0, :]
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                cv2.imwrite(
                    os.path.join(cfg.output_dir, img_name + f"_{target_category}.jpg"),
                    visualization,
                )


if __name__ == "__main__":
    run_app()
