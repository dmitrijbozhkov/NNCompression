from dataset.datasets_base import Dataset
from torchvision.models import resnet18, resnet50
from models.resnet import resnet18
from models.models_lenet import (
    LeNet3_2,
    LeNet3,
    LeNet3_3,
    LeNet3_4,
    LeNet_4,
    LeNet_5
)
import timm

def get_model(dataset: Dataset, runner_config: dict):
    if runner_config["model"] == "LeNet3":
        net = LeNet3()
    if runner_config["model"] == "LeNet3_2":
        net = LeNet3_2()
    if runner_config["model"] == "LeNet3_3":
        net = LeNet3_3()
    if runner_config["model"] == "LeNet3_4":
        net = LeNet3_4()
    if runner_config["model"] == "LeNet_5":
        net = LeNet_5()
    if runner_config["model"] == "LeNet_4":
        net = LeNet_4()
    if runner_config["model"] == "alexnet":
        net = AlexNet()
    if runner_config["model"] == "resnet18":
        # net = resnet18(
            # num_classes=dataset.output_num
        # )
        # net = timm.create_model("renset18", num_classes=dataset.output_num)
        net = resnet18(num_classes=dataset.output_num)
    if runner_config["model"] == "resnet34":
        net = resnet18(
            num_classes=dataset.output_num
        )
    if runner_config["model"] == "resnet50":
        net = resnet50(
            num_classes=dataset.output_num
        )

    return net

