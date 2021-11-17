import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import multiprocessing, cuda
from torch.backends import cudnn
from torch.utils.data import DataLoader

import voc12.dataloader
from misc import torchutils, imutils
from models import initialize_model
from models.pipeline import ModelMode, ProcessMode

cudnn.enabled = True


def _work(process_id, model, dataset, cfg):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(
        databin, shuffle=False, num_workers=cfg.num_workers // n_gpus, pin_memory=False
    )

    with torch.no_grad(), cuda.device(process_id):

        model.cuda()

        for iter, pack in enumerate(data_loader):

            img_name = pack["name"][0]
            label = pack["label"][0]
            size = pack["size"]

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            outputs = [
                model(
                    img[0].cuda(non_blocking=True),
                    model_mode=ModelMode.classification,
                    mode=ProcessMode.infer,
                )["cams"]
                for img in pack["img"]
            ]

            strided_cam = torch.sum(
                torch.stack(
                    [
                        F.interpolate(
                            torch.unsqueeze(o, 0),
                            strided_size,
                            mode="bilinear",
                            align_corners=False,
                        )[0]
                        for o in outputs
                    ]
                ),
                0,
            )

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

            valid_cat = torch.nonzero(label)[:, 0]

            strided_cam = strided_cam[valid_cat]
            strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5

            highres_cam = highres_cam[valid_cat]
            highres_cam /= F.adaptive_max_pool2d(highres_cam, (1, 1)) + 1e-5

            # save cams
            np.save(
                os.path.join(cfg.output_dir, img_name + ".npy"),
                {
                    "keys": valid_cat,
                    "cam": strided_cam.cpu(),
                    "high_res": highres_cam.cpu().numpy(),
                },
            )

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end="")


@hydra.main(config_path="./conf/irnet", config_name="make_cam")
def run_app(cfg: DictConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device("cpu" if not torch.cuda.is_available() else cfg.device)

    model = initialize_model(cfg)
    model.load_state_dict(torch.load(cfg.weights), strict=True)
    model.to(device)
    model.eval()

    n_gpus = torch.cuda.device_count()
    scales = [float(i) for i in str(cfg.scales).split("-")]
    print("Scales", scales)
    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(
        cfg.train_list, voc12_root=cfg.voc12_root, scales=scales
    )
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print("[ ", end="")
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, cfg), join=True)
    print("]")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    run_app()
