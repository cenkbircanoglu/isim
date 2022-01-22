import os
import random

import numpy as np
import torch
from PIL import Image
from chainercv.utils import read_label
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from misc.imutils import crf_inference_label
from voc12 import custom_transforms
from voc12.dataloader import (
    load_img_name_list,
    decode_int_filename,
    get_img_path,
    load_image_label_list_from_npy,
)


def rescale_tensor(img, scale):
    h, w = img.shape[-2:]
    h = int(h * scale)
    w = int(w * scale)
    rgb_rescale = transforms.Resize((h, w), interpolation=InterpolationMode.BILINEAR)
    return rgb_rescale(img)


def random_resize_long(imgs, min_long, max_long):
    target_long = random.randint(min_long, max_long)
    h, w = imgs.shape[-2:]
    if w < h:
        scale = target_long / h
    else:
        scale = target_long / w
    h = int(h * scale)
    w = int(w * scale)
    rgb_resize = transforms.Resize((h, w), interpolation=InterpolationMode.BILINEAR)
    grayscale_resize = transforms.Resize(
        (h, w), interpolation=InterpolationMode.NEAREST
    )
    if len(imgs) == 2:

        return rgb_resize(imgs[0]), grayscale_resize(imgs[1])
    else:
        return rgb_resize(imgs)


class VOC12ImageDataset(Dataset):
    def __init__(self, img_name_list_path, voc12_root, transform=None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img = Image.open(get_img_path(name_str, self.voc12_root))
        w, h = img.size
        if self.transform:
            img = self.transform(img)

        return {"name": name_str, "img": img, "idx": idx, "size": (h, w)}


class VOC12ClassificationDataset(VOC12ImageDataset):
    def __init__(self, img_name_list_path, voc12_root, transform=None):
        super().__init__(img_name_list_path, voc12_root, transform=transform)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        out["label"] = torch.from_numpy(self.label_list[idx])
        return out


class VOC12ClassificationDatasetMSF(VOC12ClassificationDataset):
    def __init__(
        self,
        img_name_list_path,
        voc12_root,
        transform=None,
        scales=(1.0,),
    ):
        super().__init__(img_name_list_path, voc12_root, transform=transform)
        self.scales = scales

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        ms_img_list = []
        img = out["img"]
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = rescale_tensor(img, s)

            ms_img_list.append(
                torch.cat([s_img.unsqueeze(0), s_img.flip(-1).unsqueeze(0)], axis=0)
            )

        out["img"] = ms_img_list

        return out


class VOC12SegmentationDataset(VOC12ImageDataset):
    def __init__(self, img_name_list_path, label_dir, voc12_root, transform=None):
        super().__init__(img_name_list_path, voc12_root)
        self.label_dir = label_dir
        self.transform_seg = transform

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)
        out["mask"] = Image.open(os.path.join(self.label_dir, name_str + ".png"))

        if self.transform_seg:
            out["img"], out["mask"] = self.transform_seg((out["img"], out["mask"]))

        return out


class VOC12PseudoSegmentationDataset(VOC12ClassificationDataset):
    def __init__(
        self,
        img_name_list_path,
        voc12_root,
        transform=None,
        temp_dir=None,
    ):
        super().__init__(img_name_list_path, voc12_root)
        self.temp_dir = temp_dir
        self.transform_seg = transform
        self.label_dir = os.path.join(self.voc12_root, "SegmentationClass")

    def update_cam(self, idx, cam, fg_thres=None):
        cams, keys = cam

        keys = np.pad(keys + 1, (1, 0), mode="constant")

        fg_conf_cam = np.pad(
            cams, ((1, 0), (0, 0), (0, 0)), mode="constant", constant_values=fg_thres
        )
        fg_conf_cam = np.argmax(fg_conf_cam, axis=0)

        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)
        path = os.path.join(self.temp_dir, name_str + ".png")

        img = np.asarray(Image.open(get_img_path(name_str, self.voc12_root)))
        pred = crf_inference_label(img, fg_conf_cam, n_labels=keys.shape[0])
        conf = keys[pred]

        Image.fromarray(conf.astype(np.uint8)).save(path)

    def generate_label(self, idx, img):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)
        path = os.path.join(self.temp_dir, name_str + ".png")
        if os.path.exists(path):
            return Image.open(path)
        else:
            return Image.fromarray(
                np.zeros((img.shape[0], img.shape[1])).astype(np.uint8)
            )

    def _get_label(self, name_str):
        label_path = os.path.join(self.label_dir, name_str + ".png")
        label = read_label(label_path, dtype=np.int32)
        label[label == 255] = -1

        return label

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)
        out["pseudo_mask"] = self.generate_label(idx, np.asarray(out["img"]))
        try:
            out["mask"] = Image.open(os.path.join(self.label_dir, name_str + ".png"))
        except:
            out["mask"] = out["pseudo_mask"]

        if self.transform_seg:
            out["img"], out["mask"], out["pseudo_mask"] = self.transform_seg(
                (out["img"], out["mask"], out["pseudo_mask"])
            )
            if out["pseudo_mask"].dim() == 3:
                out["pseudo_mask"] = out["pseudo_mask"].squeeze(0)
            if out["mask"].dim() == 3:
                out["mask"] = out["mask"].squeeze(0)
        return out


if __name__ == "__main__":
    crop_size = 512

    transform = transforms.Compose(
        [
            custom_transforms.CustomPILToTensor(),
            custom_transforms.CustomConvertImageDtype(torch.float),
            custom_transforms.CustomNormalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    ds = VOC12ClassificationDatasetMSF(
        "./voc12/train_aug.txt",
        voc12_root="../vision/data/raw/VOCdevkit/VOC2012",
        scales=[1.0, 2.0],
        transform=transform,
    )
    for a in ds:
        print(a)
    # transform = transforms.Compose(
    #     [
    #         custom_transforms.CustomPILToTensor(),
    #         transforms.Lambda(
    #             lambda x: custom_transforms.random_resize_long_tuple(
    #                 x, crop_size, crop_size * 2
    #             )
    #         ),
    #         custom_transforms.CustomRandomHorizontalFlip(0.5),
    #         custom_transforms.CustomRandomAdjustSharpness(sharpness_factor=2),
    #         custom_transforms.CustomRandomAutocontrast(),
    #         custom_transforms.CustomRandomEqualize(),
    #         custom_transforms.CustomRandomResizedCrop(size=(crop_size, crop_size)),
    #         custom_transforms.CustomRandomRotation(degrees=(0, 45)),
    #         custom_transforms.CustomConvertImageDtype(torch.float),
    #         custom_transforms.CustomRandomErasing(),
    #         custom_transforms.CustomNormalize(
    #             mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    #         ),
    #     ]
    # )
    # aa = VOC12PseudoSegmentationDataset(
    #     img_name_list_path="./voc12/train_aug.txt",
    #     voc12_root="../vision/data/raw/VOCdevkit/VOC2012",
    #     temp_dir="test",
    #     transform=transform,
    # )
    #
    # for i in aa:
    #     print(i["pseudo_mask"].shape)
