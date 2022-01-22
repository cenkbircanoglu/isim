import torch.nn as nn


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
        return x5
