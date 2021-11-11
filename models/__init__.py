from torch import nn

from models.classifier import Classifier, FixedBatchNorm
from models.decoder import Decoder
from models.encoder import Encoder
from models.pipeline import Pipeline
from models.resnest import resnest50, resnest101, resnest200, resnest269
from models.resnet import resnet50, resnet101, resnet152


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
    decoder_model = Decoder(
        in_channels=cfg.in_channels,
        out_ch=cfg.out_ch,
        input_shape=(cfg.crop_size, cfg.crop_size),
        dilate_version=cfg.dilate_version,
    )
    model = Pipeline(
        encoder_model=encoder_model,
        classifier_model=classifier_model,
        decoder_model=decoder_model,
    )
    return model
