from dataset.datasets_base import Dataset
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from models.resnet import resnet18_cifar, resnet50_cifar, resnet101_cifar, resnet152_cifar, resnet34_cifar
from models.models_lenet import (
    LeNet3_2,
    LeNet3,
    LeNet3_3,
    LeNet3_4,
    LeNet4,
    LeNet5,
    AlexNet
)

def get_model(dataset: Dataset, runner_config: dict):
    net = None

    # Simple models
    if runner_config["model"] == "LeNet3":
        net = LeNet3(
            num_classes=dataset.output_num
        )
    if runner_config["model"] == "LeNet3_2":
        net = LeNet3_2(
            num_classes=dataset.output_num
        )
    if runner_config["model"] == "LeNet3_3":
        net = LeNet3_3(
            num_classes=dataset.output_num
        )
    if runner_config["model"] == "LeNet3_4":
        net = LeNet3_4(
            num_classes=dataset.output_num
        )
    if runner_config["model"] == "LeNet4":
        net = LeNet4(
            num_classes=dataset.output_num
        )
    if runner_config["model"] == "LeNet5":
        net = LeNet5(
            num_classes=dataset.output_num
        )
    if runner_config["model"] == "alexnet":
        net = AlexNet(
            num_classes=dataset.output_num
        )

    if runner_config["model"] == "resnet18_imagenet":
        net = resnet18(
            num_classes=dataset.output_num
        )
    if runner_config["model"] == "resnet34_imagenet":
        net = resnet34(
            num_classes=dataset.output_num
        )
    if runner_config["model"] == "resnet50_imagenet":
        net = resnet50(
            num_classes=dataset.output_num
        )
    if runner_config["model"] == "resnet101":
        net = resnet101(
            num_classes=dataset.output_num
        )
    if runner_config["model"] == "resnet152_cifar":
        net = resnet152(
            num_classes=dataset.output_num
        )

    # Resnet for cifar
    if runner_config["model"] == "resnet18_cifar":
        net = resnet18_cifar(
            num_classes=dataset.output_num
        )
    if runner_config["model"] == "resnet34_cifar":
        net = resnet34_cifar(
            num_classes=dataset.output_num
        )
    if runner_config["model"] == "resnet50_cifar":
        net = resnet50_cifar(
            num_classes=dataset.output_num
        )
    if runner_config["model"] == "resnet101_cifar":
        net = resnet101_cifar(
            num_classes=dataset.output_num
        )
    if runner_config["model"] == "resnet152_cifar":
        net = resnet152_cifar(
            num_classes=dataset.output_num
        )


    assert net, f"No network found: {runner_config['model']}"

    return net

