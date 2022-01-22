import random

import torch
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode


def random_resize_long_tuple(imgs, min_long, max_long):
    if type(imgs) == tuple:
        target_long = random.randint(min_long, max_long)
        img = imgs[0]
        mask = imgs[1:]
        h, w = img.shape[-2:]
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
        return (rgb_resize(img),) + tuple(grayscale_resize(i) for i in mask)
    else:
        target_long = random.randint(min_long, max_long)
        h, w = imgs.shape[-2:]
        if w < h:
            scale = target_long / h
        else:
            scale = target_long / w
        h = int(h * scale)
        w = int(w * scale)
        rgb_resize = transforms.Resize((h, w), interpolation=InterpolationMode.BILINEAR)
        return rgb_resize(imgs)


class CustomPILToTensor(transforms.PILToTensor):
    def __call__(self, img):
        if type(img) == tuple:
            return tuple(F.pil_to_tensor(i) for i in img)
        return F.pil_to_tensor(img)


class CustomRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def forward(self, img):
        if type(img) == tuple:
            if torch.rand(1) < self.p:
                return tuple(F.hflip(i) for i in img)
            return img
        return super().forward(img)


class CustomRandomAdjustSharpness(transforms.RandomAdjustSharpness):
    def forward(self, img):
        if type(img) == tuple:
            if torch.rand(1).item() < self.p:
                return (F.adjust_sharpness(img[0], self.sharpness_factor),) + img[1:]
            return img
        return super().forward(img)


class CustomRandomAutocontrast(transforms.RandomAutocontrast):
    def forward(self, img):
        if type(img) == tuple:
            if torch.rand(1).item() < self.p:
                return (F.autocontrast(img[0]),) + img[1:]
            return img
        return super().forward(img)


class CustomRandomEqualize(transforms.RandomEqualize):
    def forward(self, img):
        if type(img) == tuple:
            if torch.rand(1).item() < self.p:
                return (F.equalize(img[0]),) + img[1:]
            return img
        return super().forward(img)


class CustomRandomCrop(transforms.RandomCrop):
    def forward(self, img):
        if type(img) == tuple:
            if self.padding is not None:
                img = [
                    F.pad(a, self.padding, self.fill, self.padding_mode) for a in img
                ]

            width, height = F.get_image_size(img[0])
            # pad the width if needed
            if self.pad_if_needed and width < self.size[1]:
                padding = [self.size[1] - width, 0]
                img = [F.pad(a, padding, self.fill, self.padding_mode) for a in img]
            # pad the height if needed
            if self.pad_if_needed and height < self.size[0]:
                padding = [0, self.size[0] - height]
                img = [F.pad(a, padding, self.fill, self.padding_mode) for a in img]

            i, j, h, w = self.get_params(img[0], self.size)

            # return F.crop(img, i, j, h, w)

            return (F.crop(img[0], i, j, h, w),) + tuple(
                (F.crop(a, i, j, h, w) for a in img[1:])
            )
        return super().forward(img)


class CustomRandomResizedCrop(transforms.RandomResizedCrop):
    def forward(self, img):
        if type(img) == tuple:
            i, j, h, w = self.get_params(img[0], self.scale, self.ratio)
            return (
                F.resized_crop(img[0], i, j, h, w, self.size, self.interpolation),
            ) + tuple(
                (
                    F.resized_crop(
                        a,
                        i,
                        j,
                        h,
                        w,
                        self.size,
                        interpolation=InterpolationMode.NEAREST,
                    )
                    for a in img[1:]
                )
            )
        return super().forward(img)


class CustomRandomRotation(transforms.RandomRotation):
    def forward(self, img):
        if type(img) == tuple:
            fill = self.fill
            if isinstance(img[0], Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * F.get_image_num_channels(img[0])
                else:
                    fill = [float(f) for f in fill]
            angle = self.get_params(self.degrees)

            return (
                F.rotate(
                    img[0],
                    angle,
                    interpolation=InterpolationMode.BILINEAR,
                    expand=self.expand,
                    center=self.center,
                    fill=fill,
                ),
            ) + tuple(
                (
                    F.rotate(
                        i,
                        angle,
                        interpolation=InterpolationMode.NEAREST,
                        expand=self.expand,
                        center=self.center,
                        fill=[fill[0]],
                    )
                    for i in img[1:]
                )
            )
        return super().forward(img)


class CustomConvertImageDtype(transforms.ConvertImageDtype):
    def forward(self, img):
        if type(img) == tuple:
            img = (F.convert_image_dtype(img[0], self.dtype),) + img[1:]
            return img
        return super().forward(img)


class CustomRandomErasing(transforms.RandomErasing):
    def forward(self, img):
        if type(img) == tuple:
            if torch.rand(1) < self.p:

                # cast self.value to script acceptable type
                if isinstance(self.value, (int, float)):
                    value = [
                        self.value,
                    ]
                elif isinstance(self.value, str):
                    value = None
                elif isinstance(self.value, tuple):
                    value = list(self.value)
                else:
                    value = self.value

                if value is not None and not (len(value) in (1, img[0].shape[-3])):
                    raise ValueError(
                        "If value is a sequence, it should have either a single value or "
                        "{} (number of input channels)".format(img[0].shape[-3])
                    )

                x, y, h, w, v = self.get_params(
                    img[0], scale=self.scale, ratio=self.ratio, value=value
                )
                return (F.erase(img[0], x, y, h, w, v, self.inplace),) + tuple(
                    (F.erase(a, x, y, h, w, v, self.inplace) for a in img[1:])
                )
            return img
        return super().forward(img)


class CustomNormalize(transforms.Normalize):
    def forward(self, img):
        if type(img) == tuple:
            return (F.normalize(img[0], self.mean, self.std, self.inplace),) + img[1:]

        return super().forward(img)


class CustomCenterCrop(transforms.CenterCrop):
    def forward(self, img):
        if type(img) == tuple:
            return (F.center_crop(img[0], self.size),) + tuple(
                (F.center_crop(i, self.size) for i in img[1:])
            )

        return super().forward(img)
