import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, input):
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=False,
            eps=self.eps,
        )


def gap2d(x, keepdims=False):
    return torch.mean(x, [2, 3], keepdim=keepdims)


class Classifier(nn.Module):
    def __init__(self, num_classes=20, in_channels=2048, *args, **kwargs):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.classifier = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_classes,
            kernel_size=(1, 1),
            bias=False,
        )

    def forward(self, x):
        x = gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, self.num_classes)
        return x

    def calculate_cam(self, x):
        with torch.set_grad_enabled(False):
            weights = torch.zeros_like(self.classifier.weight)
            with torch.no_grad():
                weights.set_(self.classifier.weight.detach())
            x = F.relu(F.conv2d(x, weight=weights))

            x = x[0] + x[1].flip(-1)
            return x


if __name__ == "__main__":
    model = Classifier().cuda()
    summary(model, input_size=(2048, 16, 16), device="cuda")
    x = torch.rand([4, 2048, 16, 16], device="cuda")
    y = model.forward(x)
    assert y.shape == (4, 20)
    cam = model.calculate_cam(x)
    assert cam.shape == (20, 16, 16)
