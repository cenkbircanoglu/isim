import torch
import torch.nn as nn

from models.deeplab import DeepLabHeadV3Plus


class Decoder(nn.Module):
    def __init__(
        self, out_ch=21, input_shape=(320, 320), in_channels=2048, *args, **kwargs
    ):
        super(Decoder, self).__init__()
        aspp_dilate = [6, 12, 18]
        # aspp_dilate = [12, 24, 36]

        inplanes = 2048
        low_level_planes = 256
        assert in_channels == inplanes

        self.classifier = DeepLabHeadV3Plus(
            inplanes, low_level_planes, out_ch, aspp_dilate, input_shape=input_shape
        )

    def forward(self, features):
        feature = {"out": features["x5"], "low_level": features["x2"]}
        return self.classifier(feature)


if __name__ == "__main__":
    model = Decoder(input_shape=(512, 512))

    x1 = torch.rand([2, 64, 128, 128])
    x2 = torch.rand([2, 256, 128, 128])
    x3 = torch.rand([2, 512, 64, 64])
    x4 = torch.rand([2, 1024, 32, 32])
    x5 = torch.rand([2, 2048, 16, 16])
    features = {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}
    y = model.forward(features)
    assert y.shape == (2, 21, 512, 512)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))  # 16253813
