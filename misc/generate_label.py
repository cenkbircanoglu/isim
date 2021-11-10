import torch
import torch.nn.functional as F


def generate_pseudo_label(
    model, imgs, cls_label_true, original_image_size, *args, **kwargs
):
    with torch.set_grad_enabled(False):
        cams = [model.forward_cam(img[0]) for img in imgs]
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
        return cams, keys
