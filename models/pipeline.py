from enum import Enum

import torch
from torch import nn

from models.resnet import resnet50
from models.classifier import Classifier
from models.decoder import Decoder
from models.encoder import Encoder


class ModelMode(Enum):
    classification = 1
    segmentation = 2


class ProcessMode(Enum):
    train = 1
    infer = 2


class Pipeline(nn.Module):
    def __init__(self, encoder_model, classifier_model, decoder_model, *args, **kwargs):
        super(Pipeline, self).__init__()
        self.encoder_model = encoder_model
        self.classifier_model = classifier_model
        self.decoder_model = decoder_model

    def forward(self, x, model_mode=ModelMode.classification, mode=ProcessMode.train):
        assert model_mode is not None and mode is not None
        features = self.encoder_model(x)
        d = {"cls": None, "cams": None, "seg": None}
        if model_mode == ModelMode.classification:
            if mode == ProcessMode.train:
                d["cls"] = self.classifier_model(features["x5"])
            elif mode == ProcessMode.infer:
                d["cams"] = self.classifier_model.calculate_cam(features["x5"])
        elif model_mode == ModelMode.segmentation:
            if mode == ProcessMode.train:
                d["cls"] = self.classifier_model(features["x5"])
                d["seg"] = self.decoder_model(features)
            elif mode == ProcessMode.infer:
                d["cams"] = self.classifier_model.calculate_cam(features["x5"])
                d["seg"] = self.decoder_model(features)
        return d

    def forward_cam(self, x):
        with torch.set_grad_enabled(False):
            features = self.encoder_model(x)
            return self.classifier_model.calculate_cam(features["x5"])

    def train(self, mode=True):
        self.encoder_model.train(mode)
        self.classifier_model.train(mode)
        self.decoder_model.train(mode)

    def trainable_parameters(self):
        return (
            list(self.encoder_model.parameters()),
            list(self.classifier_model.parameters()),
            list(self.decoder_model.parameters()),
        )


if __name__ == "__main__":
    backbone = resnet50(pretrained=True, strides=(2, 2, 2, 2))
    encoder = Encoder(backbone=backbone)
    classifier = Classifier()
    decoder = Decoder(input_shape=(512, 512))
    pipeline = Pipeline(
        encoder_model=encoder, classifier_model=classifier, decoder_model=decoder
    )
    pipeline.cuda()
    x = torch.rand([2, 3, 512, 512], device="cuda")
    features = pipeline.forward(x)
    assert features["cls"].shape == (2, 20)

    features = pipeline.forward(x, model_mode=ModelMode.segmentation)
    assert features["cls"].shape == (2, 20)
    assert features["seg"].shape == (2, 21, 512, 512)

    features = pipeline.forward(
        x, model_mode=ModelMode.classification, mode=ProcessMode.infer
    )
    assert features["cams"].shape == (20, 16, 16)

    features = pipeline.forward(
        x, model_mode=ModelMode.segmentation, mode=ProcessMode.infer
    )
    assert features["cams"].shape == (20, 16, 16)
    assert features["seg"].shape == (2, 21, 512, 512)

    backbone = resnet50(pretrained=True, strides=(2, 2, 2, 1))
    encoder = Encoder(backbone=backbone)
    classifier = Classifier()
    decoder = Decoder(input_shape=(512, 512))
    pipeline = Pipeline(
        encoder_model=encoder, classifier_model=classifier, decoder_model=decoder
    )
    pipeline.cuda()
    x = torch.rand([2, 3, 512, 512], device="cuda")
    features = pipeline.forward(x)
    assert features["cls"].shape == (2, 20)

    features = pipeline.forward(x, model_mode=ModelMode.segmentation)
    assert features["cls"].shape == (2, 20)
    assert features["seg"].shape == (2, 21, 512, 512)

    features = pipeline.forward(
        x, model_mode=ModelMode.classification, mode=ProcessMode.infer
    )
    assert features["cams"].shape == (20, 32, 32)

    features = pipeline.forward(
        x, model_mode=ModelMode.segmentation, mode=ProcessMode.infer
    )
    assert features["cams"].shape == (20, 32, 32)
    assert features["seg"].shape == (2, 21, 512, 512)

    backbone = resnet50(pretrained=True, strides=(2, 2, 1, 1))
    encoder = Encoder(backbone=backbone)
    classifier = Classifier()
    decoder = Decoder(input_shape=(512, 512))
    pipeline = Pipeline(
        encoder_model=encoder, classifier_model=classifier, decoder_model=decoder
    )
    pipeline.cuda()
    x = torch.rand([2, 3, 512, 512], device="cuda")
    features = pipeline.forward(x)
    assert features["cls"].shape == (2, 20)

    features = pipeline.forward(x, model_mode=ModelMode.segmentation)
    assert features["cls"].shape == (2, 20)
    assert features["seg"].shape == (2, 21, 512, 512)

    features = pipeline.forward(
        x, model_mode=ModelMode.classification, mode=ProcessMode.infer
    )
    assert features["cams"].shape == (20, 64, 64)

    features = pipeline.forward(
        x, model_mode=ModelMode.segmentation, mode=ProcessMode.infer
    )
    assert features["cams"].shape == (20, 64, 64)
    assert features["seg"].shape == (2, 21, 512, 512)
