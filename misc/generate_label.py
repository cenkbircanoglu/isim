import numpy as np
import torch
import torch.nn.functional as F


def generate_pseudo_label(
    model,
    imgs,
    cls_label_true,
    original_image_size,
    cam_order=0,
    fg_thres=0.3,
    *args,
    **kwargs
):
    with torch.set_grad_enabled(False):
        cams = [model.forward_cam(img[0]) for img in imgs]
        if type(cams[0]) == tuple:
            cams = list(zip(*cams))
            cams = cams[cam_order]
        strided_cam = [
            F.interpolate(
                torch.unsqueeze(o, 1),
                original_image_size,
                mode="bilinear",
                align_corners=False,
            )
            for o in cams
        ]
        strided_cam = torch.sum(torch.stack(strided_cam, 0), 0)[
            :, 0, : original_image_size[0], : original_image_size[1]
        ]

        valid_cat = torch.nonzero(cls_label_true, as_tuple=False)[:, 0]

        strided_cam = strided_cam[valid_cat]
        strided_cam /= F.adaptive_max_pool2d(strided_cam, (1, 1)) + 1e-5
        cams = strided_cam.cpu().numpy()
        keys = valid_cat.cpu().numpy()

        keys = np.pad(keys + 1, (1, 0), mode="constant")

        fg_conf_cam = np.pad(
            cams, ((1, 0), (0, 0), (0, 0)), mode="constant", constant_values=fg_thres
        )
        fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
        return fg_conf_cam, keys
