import sys
import os
from collections import OrderedDict

sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

import flwr as fl
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleCNN
from quantization import quantize_int8, dequantize_int8, calculate_size_bytes

class FlowerClient(fl.client.NumPyClient):

    def __init__(self):
        self.model = SimpleCNN()

        transform = transforms.ToTensor()

        trainset = datasets.MNIST(
            "./data",
            train=True,
            download=False,
            transform=transform,
        )

        self.trainloader = DataLoader(
            trainset,
            batch_size=64,
            shuffle=True,
        )
        testset = datasets.MNIST(
            "./data",
            train=False,
            download=False,
            transform=transform,
)

        self.testloader = DataLoader(
            testset,
            batch_size=64,
            shuffle=False,
)

    def get_parameters(self, config):
        parameters = [
        val.cpu().numpy()
        for _, val in self.model.state_dict().items()
    ]

        original_size = calculate_size_bytes(parameters)

        quantized_params, scales = quantize_int8(parameters)

        quantized_size = calculate_size_bytes(quantized_params)

        print(f"Original Model Size: {original_size / 1024:.2f} KB")
        print(f"Quantized Model Size: {quantized_size / 1024:.2f} KB")
        print(
        f"Communication Reduction: "
        f"{(1 - quantized_size / original_size) * 100:.2f}%"
    )

    # 注意：这里暂时返回原始参数，先只验证通信量统计
        return parameters

    def set_parameters(self, parameters):
        params_dict = zip(
            self.model.state_dict().keys(),
            parameters,
        )

        state_dict = OrderedDict(
            {
                k: torch.tensor(v)
                for k, v in params_dict
            }
        )

        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

    def fit(self, parameters, config):
        try:
            self.set_parameters(parameters)

            criterion = torch.nn.CrossEntropyLoss()

            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=0.001,
            )

            self.model.train()

            for images, labels in self.trainloader:
                optimizer.zero_grad()

                outputs = self.model(images)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                

            print("Training OK")

            return (
                self.get_parameters({}),
                len(self.trainloader.dataset),
                {},
            )

        except Exception as e:
            print("FIT ERROR:", e)
            raise

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

         for images, labels in self.testloader:

            outputs = self.model(images)

            _, predicted = torch.max(
                outputs.data,
                1,
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

        accuracy = correct / total

        print(
        f"Accuracy: {accuracy:.4f}"
    )

        return (
        0.0,
        total,
        {
            "accuracy": accuracy
        }
    )