import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torchmodels

from models.classifier import Classifier, FixedBatchNorm
from models.decoder import Decoder
from models.encoder import Encoder
from models.pipeline import Pipeline


def initialize_model(cfg):
    if cfg.backbone == "resnet50v1":
        backbone = torchmodels.resnet50(
            pretrained=True,
            replace_stride_with_dilation=(0, 0, 1),
            norm_layer=FixedBatchNorm,
        )
    elif cfg.backbone == "resnet50v2":
        backbone = torchmodels.resnet50(
            pretrained=True,
            replace_stride_with_dilation=(0, 1, 1),
            norm_layer=FixedBatchNorm,
        )
    elif cfg.backbone == "resnet101v1":
        backbone = torchmodels.resnet101(
            pretrained=True,
            replace_stride_with_dilation=(0, 0, 1),
            norm_layer=FixedBatchNorm,
        )
    elif cfg.backbone == "resnet101v2":
        backbone = torchmodels.resnet101(
            pretrained=True,
            replace_stride_with_dilation=(0, 1, 1),
            norm_layer=FixedBatchNorm,
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
    )
    model = Pipeline(
        encoder_model=encoder_model,
        classifier_model=classifier_model,
        decoder_model=decoder_model,
    )
    return model
