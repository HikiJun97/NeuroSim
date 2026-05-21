import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """Classic LeNet-5 for 1x28x28 MNIST. Only Conv2d / MaxPool2d / Linear / ReLU
    so the same CIM module replacement used by vgg8/resnet18 works without changes."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 1x28x28 -> 6x28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 6x14x14
            nn.Conv2d(6, 16, kernel_size=5),  # -> 16x10x10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 16x5x5
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)


def lenet5(pretrained=None, num_classes: int = 10):
    model = LeNet5(num_classes=num_classes)
    if pretrained is not None:
        state_dict = torch.load(pretrained, map_location="cpu")
        for key in list(state_dict.keys()):
            if "module" in key:
                state_dict[key.replace("module.", "")] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model
