import sys
import os
import time
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
from quantization import calculate_size_bytes
from data_utils import create_label_noniid_subset


class FlowerClient(fl.client.NumPyClient):
    """
    FedAvg Baseline Client under Non-IID data distribution.
    """

    def __init__(self, client_id):
        self.client_id = client_id
        self.model = SimpleCNN()

        self.device = torch.device("cpu")
        self.model.to(self.device)

        transform = transforms.ToTensor()

        full_trainset = datasets.MNIST(
            "./data",
            train=True,
            download=False,
            transform=transform,
        )

        trainset, labels = create_label_noniid_subset(
            full_trainset,
            client_id,
        )

        print(
            f"[Baseline Non-IID] Client {self.client_id} | "
            f"Labels: {labels} | "
            f"Samples: {len(trainset)}"
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
        return [
            val.cpu().numpy().astype("float32")
            for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters):
        params_dict = zip(
            self.model.state_dict().keys(),
            parameters,
        )

        state_dict = OrderedDict(
            {
                k: torch.tensor(v, dtype=torch.float32)
                for k, v in params_dict
            }
        )

        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        train_start_time = time.time()

        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
        )

        self.model.train()

        running_loss = 0.0

        for images, labels in self.trainloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            outputs = self.model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_time = time.time() - train_start_time
        train_loss = running_loss / len(self.trainloader)

        updated_parameters = self.get_parameters({})

        model_size_bytes = calculate_size_bytes(updated_parameters)

        upload_comm_bytes = model_size_bytes
        download_comm_bytes = model_size_bytes
        total_comm_bytes = upload_comm_bytes + download_comm_bytes

        print(
            f"[Baseline Non-IID] Client {self.client_id} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train time: {train_time:.2f}s | "
            f"Total comm: {total_comm_bytes / 1024:.2f} KB"
        )

        return (
            updated_parameters,
            len(self.trainloader.dataset),
            {
                "method": "baseline_noniid",
                "client_id": self.client_id,
                "train_loss": float(train_loss),
                "train_time": float(train_time),
                "upload_comm_bytes": int(upload_comm_bytes),
                "download_comm_bytes": int(download_comm_bytes),
                "total_comm_bytes": int(total_comm_bytes),
            },
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        criterion = torch.nn.CrossEntropyLoss()

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.testloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        test_loss = total_loss / len(self.testloader)
        accuracy = correct / total

        return (
            test_loss,
            total,
            {
                "method": "baseline_noniid",
                "client_id": self.client_id,
                "accuracy": float(accuracy),
                "test_loss": float(test_loss),
            },
        )