from torch import nn

from cam_alternatives.encoder import Encoder
from models.classifier import Classifier
from models.resnest import resnest50, resnest101, resnest200, resnest269
from models.resnet import resnet50, resnet101, resnet152


class Pipeline(nn.Module):
    def __init__(self, encoder_model, classifier_model, *args, **kwargs):
        super(Pipeline, self).__init__()
        self.encoder_model = encoder_model
        self.classifier_model = classifier_model

    def forward(self, x):
        x = self.encoder_model(x)
        return self.classifier_model(x)


def initialize_model(cfg):
    if cfg.backbone == "resnet50v1":
        backbone = resnet50(pretrained=True, strides=(2, 2, 2, 2))
    elif cfg.backbone == "resnet50v2":
        backbone = resnet50(pretrained=True, strides=(2, 2, 2, 1))
    elif cfg.backbone == "resnet50v3":
        backbone = resnet50(pretrained=True, strides=(2, 2, 1, 1))
    elif cfg.backbone == "resnet50v4":
        backbone = resnet50(pretrained=True, strides=(2, 1, 1, 1))
    elif cfg.backbone == "resnet101v1":
        backbone = resnet101(pretrained=True, strides=(2, 2, 2, 2))
    elif cfg.backbone == "resnet101v2":
        backbone = resnet101(pretrained=True, strides=(2, 2, 2, 1))
    elif cfg.backbone == "resnet101v3":
        backbone = resnet101(pretrained=True, strides=(2, 2, 1, 1))
    elif cfg.backbone == "resnet152v1":
        backbone = resnet152(pretrained=True, strides=(2, 2, 2, 2))
    elif cfg.backbone == "resnet152v2":
        backbone = resnet152(pretrained=True, strides=(2, 2, 2, 1))
    elif cfg.backbone == "resnet152v3":
        backbone = resnet152(pretrained=True, strides=(2, 2, 1, 1))
    elif cfg.backbone == "resnest50":
        dilation, dilated = 2, False
        norm_layer = nn.BatchNorm2d  # FixedBatchNorm
        backbone = resnest50(
            pretrained=True, dilation=dilation, dilated=dilated, norm_layer=norm_layer
        )
    elif cfg.backbone == "resnest101":
        dilation, dilated = 2, False
        norm_layer = nn.BatchNorm2d  # FixedBatchNorm
        backbone = resnest101(
            pretrained=True, dilation=dilation, dilated=dilated, norm_layer=norm_layer
        )
    elif cfg.backbone == "resnest200":
        dilation, dilated = 2, False
        norm_layer = nn.BatchNorm2d  # FixedBatchNorm
        backbone = resnest200(
            pretrained=True, dilation=dilation, dilated=dilated, norm_layer=norm_layer
        )
    elif cfg.backbone == "resnest269":
        dilation, dilated = 2, False
        norm_layer = nn.BatchNorm2d  # FixedBatchNorm
        backbone = resnest269(
            pretrained=True, dilation=dilation, dilated=dilated, norm_layer=norm_layer
        )
    else:
        raise Exception({"message": "fix backbone"})
    encoder_model = Encoder(backbone=backbone)
    classifier_model = Classifier(
        in_channels=cfg.in_channels, num_classes=cfg.num_classes
    )

    model = Pipeline(encoder_model=encoder_model, classifier_model=classifier_model)
    return model
