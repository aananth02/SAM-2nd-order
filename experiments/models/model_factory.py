import torchvision.models as models
from torch import nn

from wrn import WideResNet


def get_resnet18(num_classes=10, pretrained=False):
    if pretrained:
        resnet = models.resnet18(weights="DEFAULT")
    else:
        resnet = models.resnet18(weights=None)
    if num_classes != resnet.fc.out_features:
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet


def get_resnet34(num_classes=10, pretrained=False):
    if pretrained:
        resnet = models.resnet34(weights="DEFAULT")
    else:
        resnet = models.resnet34(weights=None)
    if num_classes != resnet.fc.out_features:
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet


def get_resnet50(num_classes=10, pretrained=False):
    if pretrained:
        resnet = models.resnet50(weights="DEFAULT")
    else:
        resnet = models.resnet50(weights=None)
    if num_classes != resnet.fc.out_features:
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet

def get_resnet101(num_classes=10, pretrained=False):
    if pretrained:
        resnet = models.resnet101(weights="DEFAULT")
    else:
        resnet = models.resnet101(weights=None)
    if num_classes != resnet.fc.out_features:
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    return resnet


MODEL_GETTERS = {
    "resnet18": get_resnet18,
    "resnet34": get_resnet34,
    "resnet50": get_resnet50,
    "resnet101": get_resnet101,
}
