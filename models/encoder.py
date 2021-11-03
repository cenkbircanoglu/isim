import torch
import torch.nn as nn

from torchsummary import summary
from torchvision import models as torchmodels


class Encoder(nn.Module):
    def __init__(self, backbone=None, *args, **kwargs):
        super(Encoder, self).__init__()
        self.backbone = backbone

        self.stage1 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
        )
        self.stage2 = nn.Sequential(self.backbone.layer1)
        self.stage3 = nn.Sequential(self.backbone.layer2)
        self.stage4 = nn.Sequential(self.backbone.layer3)
        self.stage5 = nn.Sequential(self.backbone.layer4)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        features = {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}
        return features


if __name__ == "__main__":
    backbone = torchmodels.resnet50(
        pretrained=True, replace_stride_with_dilation=(0, 0, 0)
    )
    model = Encoder(backbone=backbone).cuda()
    summary(model, input_size=(3, 512, 512), device="cuda")
    x = torch.rand([2, 3, 512, 512], device="cuda")
    features = model.forward(x)
    assert features["x1"].shape == (2, 64, 128, 128)
    assert features["x2"].shape == (2, 256, 128, 128)
    assert features["x3"].shape == (2, 512, 64, 64)
    assert features["x4"].shape == (2, 1024, 32, 32)
    assert features["x5"].shape == (2, 2048, 16, 16)

    backbone = torchmodels.resnet50(
        pretrained=True, replace_stride_with_dilation=(0, 0, 1)
    )
    model = Encoder(backbone=backbone).cuda()
    summary(model, input_size=(3, 512, 512), device="cuda")
    x = torch.rand([2, 3, 512, 512], device="cuda")
    features = model.forward(x)
    assert features["x1"].shape == (2, 64, 128, 128)
    assert features["x2"].shape == (2, 256, 128, 128)
    assert features["x3"].shape == (2, 512, 64, 64)
    assert features["x4"].shape == (2, 1024, 32, 32)
    assert features["x5"].shape == (2, 2048, 32, 32)

    backbone = torchmodels.resnet50(
        pretrained=True, replace_stride_with_dilation=(0, 1, 1)
    )
    model = Encoder(backbone=backbone).cuda()
    summary(model, input_size=(3, 512, 512), device="cuda")
    x = torch.rand([2, 3, 512, 512], device="cuda")
    features = model.forward(x)
    assert features["x1"].shape == (2, 64, 128, 128)
    assert features["x2"].shape == (2, 256, 128, 128)
    assert features["x3"].shape == (2, 512, 64, 64)
    assert features["x4"].shape == (2, 1024, 64, 64)
    assert features["x5"].shape == (2, 2048, 64, 64)

    backbone = torchmodels.resnet50(
        pretrained=True, replace_stride_with_dilation=(1, 1, 1)
    )
    model = Encoder(backbone=backbone).cuda()
    summary(model, input_size=(3, 512, 512), device="cuda")
    x = torch.rand([2, 3, 512, 512], device="cuda")
    features = model.forward(x)
    assert features["x1"].shape == (2, 64, 128, 128)
    assert features["x2"].shape == (2, 256, 128, 128)
    assert features["x3"].shape == (2, 512, 128, 128)
    assert features["x4"].shape == (2, 1024, 128, 128)
    assert features["x5"].shape == (2, 2048, 128, 128)

    backbone = torchmodels.resnet50(
        pretrained=True, replace_stride_with_dilation=(1, 0, 1)
    )
    model = Encoder(backbone=backbone).cuda()
    summary(model, input_size=(3, 512, 512), device="cuda")
    x = torch.rand([2, 3, 512, 512], device="cuda")
    features = model.forward(x)
    assert features["x1"].shape == (2, 64, 128, 128)
    assert features["x2"].shape == (2, 256, 128, 128)
    assert features["x3"].shape == (2, 512, 128, 128)
    assert features["x4"].shape == (2, 1024, 64, 64)
    assert features["x5"].shape == (2, 2048, 64, 64)

    backbone = torchmodels.resnet101(
        pretrained=True, replace_stride_with_dilation=(0, 0, 0)
    )
    model = Encoder(backbone=backbone).cuda()
    summary(model, input_size=(3, 512, 512), device="cuda")
    x = torch.rand([2, 3, 512, 512], device="cuda")
    features = model.forward(x)
    assert features["x1"].shape == (2, 64, 128, 128)
    assert features["x2"].shape == (2, 256, 128, 128)
    assert features["x3"].shape == (2, 512, 64, 64)
    assert features["x4"].shape == (2, 1024, 32, 32)
    assert features["x5"].shape == (2, 2048, 16, 16)
