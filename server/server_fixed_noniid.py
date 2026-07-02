import os
import csv
import sys
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np

from flwr.common import (
    FitRes,
    EvaluateRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.server.client_proxy import ClientProxy

sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

from model import SimpleCNN
from quantization import (
    dequantize_int8_from_flower,
    deserialize_scales,
)


NUM_CLIENTS = 5
NUM_ROUNDS = 10

RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "fixed_noniid_results.csv"
)


def get_initial_parameters():
    model = SimpleCNN()

    parameters = [
        val.cpu().numpy().astype(np.float32)
        for _, val in model.state_dict().items()
    ]

    return ndarrays_to_parameters(parameters)


class QuantizedFedAvg(fl.server.strategy.FedAvg):
    """
    Fixed INT8 Quantization FedAvg.

    Client upload:
        quantized int8 values carried as float32 arrays
        scales sent through metrics

    Server:
        receives quantized values
        restores int8
        dequantizes to float32
        performs weighted FedAvg
        sends float32 global model back to clients
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        os.makedirs(
            RESULTS_DIR,
            exist_ok=True,
        )

        with open(
            RESULTS_FILE,
            mode="w",
            newline="",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    "round",
                    "method",
                    "num_clients",
                    "total_examples",
                    "accuracy",
                    "test_loss",
                    "avg_train_loss",
                    "avg_train_time",
                    "upload_comm_bytes",
                    "download_comm_bytes",
                    "total_comm_bytes",
                    "avg_reduction",
                    "upload_comm_kb",
                    "download_comm_kb",
                    "total_comm_kb",
                ]
            )

        self.fit_metrics_by_round = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        if failures:
            print(
                f"[Fixed INT8] Round {server_round} "
                f"fit failures: {len(failures)}"
            )

        dequantized_results = []

        total_examples = 0
        upload_comm_bytes = 0
        download_comm_bytes = 0
        total_comm_bytes = 0

        weighted_train_loss = 0.0
        weighted_train_time = 0.0
        weighted_reduction = 0.0

        for _, fit_res in results:
            flower_quantized_params = parameters_to_ndarrays(
                fit_res.parameters
            )

            scales = deserialize_scales(
                fit_res.metrics["scales"]
            )

            dequantized_params = dequantize_int8_from_flower(
                flower_quantized_params,
                scales,
            )

            num_examples = fit_res.num_examples
            total_examples += num_examples

            dequantized_results.append(
                (
                    dequantized_params,
                    num_examples,
                )
            )

            upload_comm_bytes += int(
                fit_res.metrics.get(
                    "upload_comm_bytes",
                    0,
                )
            )

            download_comm_bytes += int(
                fit_res.metrics.get(
                    "download_comm_bytes",
                    0,
                )
            )

            total_comm_bytes += int(
                fit_res.metrics.get(
                    "total_comm_bytes",
                    0,
                )
            )

            weighted_train_loss += (
                float(
                    fit_res.metrics.get(
                        "train_loss",
                        0.0,
                    )
                )
                * num_examples
            )

            weighted_train_time += (
                float(
                    fit_res.metrics.get(
                        "train_time",
                        0.0,
                    )
                )
                * num_examples
            )

            weighted_reduction += (
                float(
                    fit_res.metrics.get(
                        "communication_reduction",
                        0.0,
                    )
                )
                * num_examples
            )

        aggregated_parameters = self.weighted_average(
            dequantized_results
        )

        aggregated_parameters_flower = ndarrays_to_parameters(
            aggregated_parameters
        )

        avg_train_loss = (
            weighted_train_loss / total_examples
            if total_examples > 0
            else 0.0
        )

        avg_train_time = (
            weighted_train_time / total_examples
            if total_examples > 0
            else 0.0
        )

        avg_reduction = (
            weighted_reduction / total_examples
            if total_examples > 0
            else 0.0
        )

        self.fit_metrics_by_round[server_round] = {
            "round": server_round,
            "method": "fixed_int8",
            "num_clients": len(results),
            "total_examples": total_examples,
            "avg_train_loss": avg_train_loss,
            "avg_train_time": avg_train_time,
            "upload_comm_bytes": upload_comm_bytes,
            "download_comm_bytes": download_comm_bytes,
            "total_comm_bytes": total_comm_bytes,
            "avg_reduction": avg_reduction,
        }

        print()
        print("=" * 70)
        print(f"[Fixed INT8] Round {server_round} Fit Aggregation")
        print(f"Clients: {len(results)}")
        print(f"Total examples: {total_examples}")
        print(f"Average train loss: {avg_train_loss:.4f}")
        print(f"Average train time: {avg_train_time:.2f}s")
        print(f"Upload communication: {upload_comm_bytes / 1024:.2f} KB")
        print(f"Download communication: {download_comm_bytes / 1024:.2f} KB")
        print(f"Total communication: {total_comm_bytes / 1024:.2f} KB")
        print(f"Average upload reduction: {avg_reduction:.2f}%")
        print("=" * 70)
        print()

        return aggregated_parameters_flower, {
            "avg_train_loss": avg_train_loss,
            "avg_train_time": avg_train_time,
            "upload_comm_bytes": upload_comm_bytes,
            "download_comm_bytes": download_comm_bytes,
            "total_comm_bytes": total_comm_bytes,
            "avg_reduction": avg_reduction,
        }

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:

        if not results:
            return None, {}

        if failures:
            print(
                f"[Fixed INT8] Round {server_round} "
                f"evaluate failures: {len(failures)}"
            )

        total_examples = sum(
            evaluate_res.num_examples
            for _, evaluate_res in results
        )

        weighted_loss = sum(
            evaluate_res.loss * evaluate_res.num_examples
            for _, evaluate_res in results
        )

        test_loss = weighted_loss / total_examples

        weighted_accuracy = 0.0

        for _, evaluate_res in results:
            weighted_accuracy += (
                float(
                    evaluate_res.metrics.get(
                        "accuracy",
                        0.0,
                    )
                )
                * evaluate_res.num_examples
            )

        accuracy = weighted_accuracy / total_examples

        fit_metrics = self.fit_metrics_by_round.get(
            server_round,
            {},
        )

        upload_comm_bytes = fit_metrics.get(
            "upload_comm_bytes",
            0,
        )

        download_comm_bytes = fit_metrics.get(
            "download_comm_bytes",
            0,
        )

        total_comm_bytes = fit_metrics.get(
            "total_comm_bytes",
            0,
        )

        with open(
            RESULTS_FILE,
            mode="a",
            newline="",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    server_round,
                    "fixed_int8",
                    fit_metrics.get(
                        "num_clients",
                        len(results),
                    ),
                    fit_metrics.get(
                        "total_examples",
                        total_examples,
                    ),
                    accuracy,
                    test_loss,
                    fit_metrics.get(
                        "avg_train_loss",
                        0.0,
                    ),
                    fit_metrics.get(
                        "avg_train_time",
                        0.0,
                    ),
                    upload_comm_bytes,
                    download_comm_bytes,
                    total_comm_bytes,
                    fit_metrics.get(
                        "avg_reduction",
                        0.0,
                    ),
                    upload_comm_bytes / 1024,
                    download_comm_bytes / 1024,
                    total_comm_bytes / 1024,
                ]
            )

        print()
        print("=" * 70)
        print(f"[Fixed INT8] Round {server_round} Evaluation")
        print(f"Global accuracy: {accuracy:.4f}")
        print(f"Global test loss: {test_loss:.4f}")
        print(f"Round total communication: {total_comm_bytes / 1024:.2f} KB")
        print("=" * 70)
        print()

        return test_loss, {
            "accuracy": accuracy,
            "test_loss": test_loss,
        }

    @staticmethod
    def weighted_average(
        results: List[Tuple[List[np.ndarray], int]]
    ) -> List[np.ndarray]:

        total_examples = sum(
            num_examples
            for _, num_examples in results
        )

        num_layers = len(results[0][0])

        aggregated_parameters = []

        for layer_idx in range(num_layers):
            layer_sum = np.zeros_like(
                results[0][0][layer_idx],
                dtype=np.float32,
            )

            for parameters, num_examples in results:
                layer_sum += (
                    parameters[layer_idx].astype(np.float32)
                    * num_examples
                )

            aggregated_layer = layer_sum / total_examples

            aggregated_parameters.append(
                aggregated_layer.astype(np.float32)
            )

        return aggregated_parameters


strategy = QuantizedFedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    initial_parameters=get_initial_parameters(),
)


print()
print("=" * 70)
print("Fixed INT8 Non-IID Server")
print("Dataset: MNIST")
print(f"Clients: {NUM_CLIENTS}")
print(f"Rounds: {NUM_ROUNDS}")
print("Client upload: INT8 quantized values")
print("Scales: sent through Flower metrics")
print("Server aggregation: dequantize -> FedAvg")
print("Server download: float32 global model")
print(f"Results file: {RESULTS_FILE}")
print("=" * 70)
print()


fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(
        num_rounds=NUM_ROUNDS,
    ),
    strategy=strategy,
)