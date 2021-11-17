import os

import hydra
import imageio
import numpy as np
from omegaconf import DictConfig
from torch import multiprocessing
from torch.utils.data import DataLoader

import voc12.dataloader
from misc import torchutils, imutils


def _work(process_id, infer_dataset, cfg):
    databin = infer_dataset[process_id]
    infer_data_loader = DataLoader(
        databin, shuffle=False, num_workers=0, pin_memory=False
    )

    for iter, pack in enumerate(infer_data_loader):
        img_name = voc12.dataloader.decode_int_filename(pack["name"][0])
        img = pack["img"][0].numpy()
        cam_dict = np.load(
            os.path.join(cfg.cam_out_dir, img_name + ".npy"), allow_pickle=True
        ).item()

        cams = cam_dict["high_res"]
        keys = np.pad(cam_dict["keys"] + 1, (1, 0), mode="constant")

        # 1. find confident fg & bg
        fg_conf_cam = np.pad(
            cams,
            ((1, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=cfg.conf_fg_thres,
        )
        fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
        pred = imutils.crf_inference_label(img, fg_conf_cam, n_labels=keys.shape[0])
        fg_conf = keys[pred]

        bg_conf_cam = np.pad(
            cams,
            ((1, 0), (0, 0), (0, 0)),
            mode="constant",
            constant_values=cfg.conf_bg_thres,
        )
        bg_conf_cam = np.argmax(bg_conf_cam, axis=0)
        pred = imutils.crf_inference_label(img, bg_conf_cam, n_labels=keys.shape[0])
        bg_conf = keys[pred]

        # 2. combine confident fg & bg
        conf = fg_conf.copy()
        conf[fg_conf == 0] = 255
        conf[bg_conf + fg_conf == 0] = 0

        imageio.imwrite(
            os.path.join(cfg.output_dir, img_name + ".png"),
            conf.astype(np.uint8),
        )

        if process_id == cfg.num_workers - 1 and iter % (len(databin) // 20) == 0:
            print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end="")


@hydra.main(config_path="./conf/irnet", config_name="cam_to_ir_label")
def run_app(cfg: DictConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    dataset = voc12.dataloader.VOC12ImageDataset(
        cfg.train_list, voc12_root=cfg.voc12_root, img_normal=None, to_torch=False
    )
    dataset = torchutils.split_dataset(dataset, cfg.num_workers)

    print("[ ", end="")
    multiprocessing.spawn(_work, nprocs=cfg.num_workers, args=(dataset, cfg), join=True)
    print("]")


if __name__ == "__main__":
    run_app()
