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
    dequantize_uniform_from_flower,
    deserialize_scales,
)


NUM_CLIENTS = 5
NUM_ROUNDS = 10

RESULTS_DIR = "results"

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "adaptive_results.csv",
)

CLIENT_DETAIL_FILE = os.path.join(
    RESULTS_DIR,
    "adaptive_client_details.csv",
)


def get_initial_parameters():
    model = SimpleCNN()

    parameters = [
        val.cpu().numpy().astype(np.float32)
        for _, val in model.state_dict().items()
    ]

    return ndarrays_to_parameters(parameters)


def fit_config(server_round: int):
    return {
        "server_round": int(server_round),
    }


class AdaptiveQuantizedFedAvg(fl.server.strategy.FedAvg):
    """
    Adaptive Quantization FedAvg.

    Client upload:
        4-bit / 6-bit / 8-bit quantized values carried as float32 arrays.
        Scales and bit-width are sent through metrics.

    Server:
        receives quantized values,
        restores client-specific bit-width,
        dequantizes to float32,
        performs weighted FedAvg,
        sends float32 global model back to clients.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        os.makedirs(
            RESULTS_DIR,
            exist_ok=True,
        )

        with open(
            SUMMARY_FILE,
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
                    "avg_bit_width",
                    "avg_network_score",
                    "avg_loss_change",
                    "avg_bandwidth_mbps",
                    "avg_latency_ms",
                    "upload_comm_bytes",
                    "download_comm_bytes",
                    "total_comm_bytes",
                    "avg_reduction",
                    "avg_estimated_upload_time",
                    "avg_estimated_total_time",
                    "upload_comm_kb",
                    "download_comm_kb",
                    "total_comm_kb",
                ]
            )

        with open(
            CLIENT_DETAIL_FILE,
            mode="w",
            newline="",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    "round",
                    "client_id",
                    "num_examples",
                    "train_loss",
                    "train_time",
                    "bandwidth_mbps",
                    "latency_ms",
                    "network_score",
                    "loss_change",
                    "bit_width",
                    "upload_comm_bytes",
                    "download_comm_bytes",
                    "total_comm_bytes",
                    "communication_reduction",
                    "estimated_upload_time",
                    "estimated_total_time",
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
                f"[Adaptive] Round {server_round} "
                f"fit failures: {len(failures)}"
            )

        dequantized_results = []

        total_examples = 0

        upload_comm_bytes = 0
        download_comm_bytes = 0
        total_comm_bytes = 0

        weighted_train_loss = 0.0
        weighted_train_time = 0.0
        weighted_bit_width = 0.0
        weighted_network_score = 0.0
        weighted_loss_change = 0.0
        weighted_bandwidth = 0.0
        weighted_latency = 0.0
        weighted_reduction = 0.0
        weighted_estimated_upload_time = 0.0
        weighted_estimated_total_time = 0.0

        client_rows = []

        for _, fit_res in results:
            flower_quantized_params = parameters_to_ndarrays(
                fit_res.parameters
            )

            metrics = fit_res.metrics

            scales = deserialize_scales(
                metrics["scales"]
            )

            bit_width = int(
                metrics["bit_width"]
            )

            dequantized_params = dequantize_uniform_from_flower(
                flower_quantized_params,
                scales,
                bit_width,
            )

            num_examples = fit_res.num_examples
            total_examples += num_examples

            dequantized_results.append(
                (
                    dequantized_params,
                    num_examples,
                )
            )

            client_id = int(
                metrics.get(
                    "client_id",
                    -1,
                )
            )

            train_loss = float(
                metrics.get(
                    "train_loss",
                    0.0,
                )
            )

            train_time = float(
                metrics.get(
                    "train_time",
                    0.0,
                )
            )

            bandwidth_mbps = float(
                metrics.get(
                    "bandwidth_mbps",
                    0.0,
                )
            )

            latency_ms = float(
                metrics.get(
                    "latency_ms",
                    0.0,
                )
            )

            network_score = float(
                metrics.get(
                    "network_score",
                    0.0,
                )
            )

            loss_change = float(
                metrics.get(
                    "loss_change",
                    0.0,
                )
            )

            client_upload_comm = int(
                metrics.get(
                    "upload_comm_bytes",
                    0,
                )
            )

            client_download_comm = int(
                metrics.get(
                    "download_comm_bytes",
                    0,
                )
            )

            client_total_comm = int(
                metrics.get(
                    "total_comm_bytes",
                    0,
                )
            )

            communication_reduction = float(
                metrics.get(
                    "communication_reduction",
                    0.0,
                )
            )

            estimated_upload_time = float(
                metrics.get(
                    "estimated_upload_time",
                    0.0,
                )
            )

            estimated_total_time = float(
                metrics.get(
                    "estimated_total_time",
                    0.0,
                )
            )

            upload_comm_bytes += client_upload_comm
            download_comm_bytes += client_download_comm
            total_comm_bytes += client_total_comm

            weighted_train_loss += train_loss * num_examples
            weighted_train_time += train_time * num_examples
            weighted_bit_width += bit_width * num_examples
            weighted_network_score += network_score * num_examples
            weighted_loss_change += loss_change * num_examples
            weighted_bandwidth += bandwidth_mbps * num_examples
            weighted_latency += latency_ms * num_examples
            weighted_reduction += communication_reduction * num_examples
            weighted_estimated_upload_time += estimated_upload_time * num_examples
            weighted_estimated_total_time += estimated_total_time * num_examples

            client_rows.append(
                [
                    server_round,
                    client_id,
                    num_examples,
                    train_loss,
                    train_time,
                    bandwidth_mbps,
                    latency_ms,
                    network_score,
                    loss_change,
                    bit_width,
                    client_upload_comm,
                    client_download_comm,
                    client_total_comm,
                    communication_reduction,
                    estimated_upload_time,
                    estimated_total_time,
                ]
            )

        aggregated_parameters = self.weighted_average(
            dequantized_results
        )

        aggregated_parameters_flower = ndarrays_to_parameters(
            aggregated_parameters
        )

        avg_train_loss = weighted_train_loss / total_examples
        avg_train_time = weighted_train_time / total_examples
        avg_bit_width = weighted_bit_width / total_examples
        avg_network_score = weighted_network_score / total_examples
        avg_loss_change = weighted_loss_change / total_examples
        avg_bandwidth_mbps = weighted_bandwidth / total_examples
        avg_latency_ms = weighted_latency / total_examples
        avg_reduction = weighted_reduction / total_examples
        avg_estimated_upload_time = weighted_estimated_upload_time / total_examples
        avg_estimated_total_time = weighted_estimated_total_time / total_examples

        self.fit_metrics_by_round[server_round] = {
            "round": server_round,
            "method": "adaptive",
            "num_clients": len(results),
            "total_examples": total_examples,
            "avg_train_loss": avg_train_loss,
            "avg_train_time": avg_train_time,
            "avg_bit_width": avg_bit_width,
            "avg_network_score": avg_network_score,
            "avg_loss_change": avg_loss_change,
            "avg_bandwidth_mbps": avg_bandwidth_mbps,
            "avg_latency_ms": avg_latency_ms,
            "upload_comm_bytes": upload_comm_bytes,
            "download_comm_bytes": download_comm_bytes,
            "total_comm_bytes": total_comm_bytes,
            "avg_reduction": avg_reduction,
            "avg_estimated_upload_time": avg_estimated_upload_time,
            "avg_estimated_total_time": avg_estimated_total_time,
        }

        with open(
            CLIENT_DETAIL_FILE,
            mode="a",
            newline="",
        ) as file:
            writer = csv.writer(file)
            writer.writerows(client_rows)

        print()
        print("=" * 70)
        print(f"[Adaptive] Round {server_round} Fit Aggregation")
        print(f"Clients: {len(results)}")
        print(f"Total examples: {total_examples}")
        print(f"Average train loss: {avg_train_loss:.4f}")
        print(f"Average train time: {avg_train_time:.2f}s")
        print(f"Average bit-width: {avg_bit_width:.2f}")
        print(f"Average network score: {avg_network_score:.4f}")
        print(f"Average bandwidth: {avg_bandwidth_mbps:.2f} Mbps")
        print(f"Average latency: {avg_latency_ms:.2f} ms")
        print(f"Upload communication: {upload_comm_bytes / 1024:.2f} KB")
        print(f"Download communication: {download_comm_bytes / 1024:.2f} KB")
        print(f"Total communication: {total_comm_bytes / 1024:.2f} KB")
        print(f"Average reduction: {avg_reduction:.2f}%")
        print(f"Average estimated upload time: {avg_estimated_upload_time:.4f}s")
        print("=" * 70)
        print()

        return aggregated_parameters_flower, {
            "avg_train_loss": avg_train_loss,
            "avg_train_time": avg_train_time,
            "avg_bit_width": avg_bit_width,
            "avg_network_score": avg_network_score,
            "avg_loss_change": avg_loss_change,
            "avg_bandwidth_mbps": avg_bandwidth_mbps,
            "avg_latency_ms": avg_latency_ms,
            "upload_comm_bytes": upload_comm_bytes,
            "download_comm_bytes": download_comm_bytes,
            "total_comm_bytes": total_comm_bytes,
            "avg_reduction": avg_reduction,
            "avg_estimated_upload_time": avg_estimated_upload_time,
            "avg_estimated_total_time": avg_estimated_total_time,
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
                f"[Adaptive] Round {server_round} "
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
            SUMMARY_FILE,
            mode="a",
            newline="",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    server_round,
                    "adaptive",
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
                    fit_metrics.get(
                        "avg_bit_width",
                        0.0,
                    ),
                    fit_metrics.get(
                        "avg_network_score",
                        0.0,
                    ),
                    fit_metrics.get(
                        "avg_loss_change",
                        0.0,
                    ),
                    fit_metrics.get(
                        "avg_bandwidth_mbps",
                        0.0,
                    ),
                    fit_metrics.get(
                        "avg_latency_ms",
                        0.0,
                    ),
                    upload_comm_bytes,
                    download_comm_bytes,
                    total_comm_bytes,
                    fit_metrics.get(
                        "avg_reduction",
                        0.0,
                    ),
                    fit_metrics.get(
                        "avg_estimated_upload_time",
                        0.0,
                    ),
                    fit_metrics.get(
                        "avg_estimated_total_time",
                        0.0,
                    ),
                    upload_comm_bytes / 1024,
                    download_comm_bytes / 1024,
                    total_comm_bytes / 1024,
                ]
            )

        print()
        print("=" * 70)
        print(f"[Adaptive] Round {server_round} Evaluation")
        print(f"Global accuracy: {accuracy:.4f}")
        print(f"Global test loss: {test_loss:.4f}")
        print(f"Average bit-width: {fit_metrics.get('avg_bit_width', 0.0):.2f}")
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


strategy = AdaptiveQuantizedFedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    initial_parameters=get_initial_parameters(),
    on_fit_config_fn=fit_config,
)


print()
print("=" * 70)
print("Adaptive Quantization Server Started")
print("Dataset: MNIST")
print(f"Clients: {NUM_CLIENTS}")
print(f"Rounds: {NUM_ROUNDS}")
print("Client upload: adaptive 4-bit / 6-bit / 8-bit quantized values")
print("Adaptive rule: network-aware + training-loss-aware")
print("Scales and bit-width: sent through Flower metrics")
print("Server aggregation: client-specific dequantization -> FedAvg")
print("Server download: float32 global model")
print(f"Summary results file: {SUMMARY_FILE}")
print(f"Client detail file: {CLIENT_DETAIL_FILE}")
print("=" * 70)
print()


fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(
        num_rounds=NUM_ROUNDS,
    ),
    strategy=strategy,
)