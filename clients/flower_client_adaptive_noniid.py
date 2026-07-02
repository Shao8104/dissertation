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
from adaptive_policy_noniid import AdaptivePolicy
from network_simulator import (
    simulate_network_condition,
    estimate_transmission_time,
)
from quantization import (
    calculate_size_bytes,
    quantize_uniform_for_flower,
    calculate_uniform_payload_size_bytes,
    calculate_uniform_reduction,
    serialize_scales,
)
from data_utils import create_label_noniid_subset


class FlowerClient(fl.client.NumPyClient):
    """
    Adaptive Quantization Client under Non-IID data distribution.
    """

    def __init__(self, client_id):
        self.client_id = client_id
        self.model = SimpleCNN()
        self.policy = AdaptivePolicy()

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
            f"[Adaptive Non-IID] Client {self.client_id} | "
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

    def get_float32_parameters(self):
        return [
            val.cpu().numpy().astype("float32")
            for _, val in self.model.state_dict().items()
        ]

    def get_parameters(self, config):
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

        server_round = int(
            config.get(
                "server_round",
                1,
            )
        )

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

        bandwidth_mbps, latency_ms = simulate_network_condition(
            client_id=self.client_id,
            server_round=server_round,
        )

        policy_result = self.policy.choose_bitwidth(
            bandwidth_mbps=bandwidth_mbps,
            latency_ms=latency_ms,
            train_loss=train_loss,
            server_round=server_round,
        )

        bit_width = int(policy_result["bit_width"])

        assert bit_width in [4, 6, 8], (
            f"Invalid bit-width: {bit_width}"
        )

        network_score = float(policy_result["network_score"])
        loss_change = float(policy_result["loss_change"])

        training_score = float(
            policy_result.get(
                "training_score",
                0.0,
            )
        )

        final_score = float(
            policy_result.get(
                "final_score",
                network_score,
            )
        )

        float32_params = self.get_float32_parameters()

        (
            flower_quantized_params,
            quantized_params,
            scales,
        ) = quantize_uniform_for_flower(
            float32_params,
            bit_width,
        )

        original_size, quantized_size, reduction = calculate_uniform_reduction(
            float32_params,
            quantized_params,
            scales,
            bit_width,
        )

        upload_comm_bytes = calculate_uniform_payload_size_bytes(
            quantized_params,
            scales,
            bit_width,
        )

        download_comm_bytes = calculate_size_bytes(
            float32_params
        )

        total_comm_bytes = upload_comm_bytes + download_comm_bytes

        estimated_upload_time = estimate_transmission_time(
            payload_bytes=upload_comm_bytes,
            bandwidth_mbps=bandwidth_mbps,
            latency_ms=latency_ms,
        )

        estimated_total_time = train_time + estimated_upload_time

        print(
            f"[Adaptive Non-IID] Client {self.client_id} | "
            f"Round: {server_round} | "
            f"Train loss: {train_loss:.4f} | "
            f"Bit-width: {bit_width} | "
            f"Bandwidth: {bandwidth_mbps:.2f} Mbps | "
            f"Latency: {latency_ms:.2f} ms | "
            f"Network score: {network_score:.4f} | "
            f"Training score: {training_score:.4f} | "
            f"Final score: {final_score:.4f} | "
            f"Upload: {upload_comm_bytes / 1024:.2f} KB"
        )

        return (
            flower_quantized_params,
            len(self.trainloader.dataset),
            {
                "method": "adaptive_noniid",
                "client_id": int(self.client_id),
                "server_round": int(server_round),
                "train_loss": float(train_loss),
                "train_time": float(train_time),
                "bandwidth_mbps": float(bandwidth_mbps),
                "latency_ms": float(latency_ms),
                "network_score": float(network_score),
                "training_score": float(training_score),
                "final_score": float(final_score),
                "loss_change": float(loss_change),
                "bit_width": int(bit_width),
                "original_size_bytes": int(original_size),
                "upload_comm_bytes": int(upload_comm_bytes),
                "download_comm_bytes": int(download_comm_bytes),
                "total_comm_bytes": int(total_comm_bytes),
                "communication_reduction": float(reduction),
                "estimated_upload_time": float(estimated_upload_time),
                "estimated_total_time": float(estimated_total_time),
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
                "method": "adaptive_noniid",
                "client_id": int(self.client_id),
                "accuracy": float(accuracy),
                "test_loss": float(test_loss),
            },
        )