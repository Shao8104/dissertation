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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import SimpleCNN
from quantization import (
    calculate_size_bytes,
    calculate_int8_payload_size_bytes,
    calculate_reduction,
    quantize_int8_for_flower,
    serialize_scales,
)


class FlowerClient(fl.client.NumPyClient):
    """
    Fixed INT8 Quantization Client.

    Server sends float32 global model.
    Client trains locally in float32.
    Client quantizes updated parameters to int8.
    Client uploads quantized values using Flower-compatible float32 carrier.
    Scales are sent through metrics.
    Server dequantizes before FedAvg aggregation.
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

        num_clients = 5
        subset_size = len(full_trainset) // num_clients

        start_idx = client_id * subset_size

        if client_id == num_clients - 1:
            end_idx = len(full_trainset)
        else:
            end_idx = start_idx + subset_size

        trainset = Subset(
            full_trainset,
            range(start_idx, end_idx),
        )

        print(
            f"[Fixed INT8] Client {self.client_id}: "
            f"training samples {start_idx} - {end_idx}"
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

    def get_float32_parameters(self):
        return [
            val.cpu().numpy().astype("float32")
            for _, val in self.model.state_dict().items()
        ]

    def get_parameters(self, config):
        """
        Used only if server requests initial parameters.
        Fixed experiment should normally use server-side initial_parameters.
        """

        return self.get_float32_parameters()

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

        float32_params = self.get_float32_parameters()

        (
            flower_quantized_params,
            int8_params,
            scales,
        ) = quantize_int8_for_flower(float32_params)

        original_size, quantized_size, reduction = calculate_reduction(
            float32_params,
            int8_params,
            scales,
        )

        upload_comm_bytes = calculate_int8_payload_size_bytes(
            int8_params,
            scales,
        )

        download_comm_bytes = calculate_size_bytes(float32_params)

        total_comm_bytes = upload_comm_bytes + download_comm_bytes

        print(
            f"[Fixed INT8] Client {self.client_id} | "
            f"Training OK | "
            f"Train loss: {train_loss:.4f} | "
            f"Train time: {train_time:.2f}s | "
            f"Original: {original_size / 1024:.2f} KB | "
            f"INT8 upload: {upload_comm_bytes / 1024:.2f} KB | "
            f"Reduction: {reduction:.2f}% | "
            f"Download: {download_comm_bytes / 1024:.2f} KB | "
            f"Total comm: {total_comm_bytes / 1024:.2f} KB"
        )

        return (
            flower_quantized_params,
            len(self.trainloader.dataset),
            {
                "method": "fixed_int8",
                "client_id": self.client_id,
                "train_loss": float(train_loss),
                "train_time": float(train_time),
                "original_size_bytes": int(original_size),
                "upload_comm_bytes": int(upload_comm_bytes),
                "download_comm_bytes": int(download_comm_bytes),
                "total_comm_bytes": int(total_comm_bytes),
                "communication_reduction": float(reduction),
                "scales": serialize_scales(scales),
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

                _, predicted = torch.max(
                    outputs.data,
                    1,
                )

                total += labels.size(0)

                correct += (
                    predicted == labels
                ).sum().item()

        test_loss = total_loss / len(self.testloader)
        accuracy = correct / total

        print(
            f"[Fixed INT8] Client {self.client_id} | "
            f"Test loss: {test_loss:.4f} | "
            f"Accuracy: {accuracy:.4f}"
        )

        return (
            test_loss,
            total,
            {
                "method": "fixed_int8",
                "client_id": self.client_id,
                "accuracy": float(accuracy),
                "test_loss": float(test_loss),
            },
        )