import os
import csv

import flwr as fl


NUM_CLIENTS = 5
NUM_ROUNDS = 10

RESULT_DIR = "results"
RESULT_FILE = os.path.join(
    RESULT_DIR,
    "baseline_noniid_results.csv",
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True,
)

with open(
    RESULT_FILE,
    mode="w",
    newline="",
) as f:
    writer = csv.writer(f)

    writer.writerow(
        [
            "round",
            "accuracy",
            "loss",
            "avg_train_loss",
            "avg_train_time",
            "upload_comm_kb",
            "download_comm_kb",
            "total_comm_kb",
        ]
    )


current_round = 0

latest_fit_metrics = {
    "train_loss": 0.0,
    "train_time": 0.0,
    "upload_comm": 0.0,
    "download_comm": 0.0,
    "total_comm": 0.0,
}


def fit_metrics_aggregation_fn(metrics):
    global latest_fit_metrics

    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    train_loss = sum(
        num_examples * m["train_loss"]
        for num_examples, m in metrics
    ) / total_examples

    train_time = sum(
        num_examples * m["train_time"]
        for num_examples, m in metrics
    ) / total_examples

    upload_comm = sum(
        m["upload_comm_bytes"]
        for _, m in metrics
    ) / 1024

    download_comm = sum(
        m["download_comm_bytes"]
        for _, m in metrics
    ) / 1024

    total_comm = sum(
        m["total_comm_bytes"]
        for _, m in metrics
    ) / 1024

    latest_fit_metrics = {
        "train_loss": train_loss,
        "train_time": train_time,
        "upload_comm": upload_comm,
        "download_comm": download_comm,
        "total_comm": total_comm,
    }

    return latest_fit_metrics


def evaluate_metrics_aggregation_fn(metrics):
    global current_round
    global latest_fit_metrics

    current_round += 1

    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    accuracy = sum(
        num_examples * m["accuracy"]
        for num_examples, m in metrics
    ) / total_examples

    loss = sum(
        num_examples * m["test_loss"]
        for num_examples, m in metrics
    ) / total_examples

    with open(
        RESULT_FILE,
        mode="a",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                current_round,
                accuracy,
                loss,
                latest_fit_metrics["train_loss"],
                latest_fit_metrics["train_time"],
                latest_fit_metrics["upload_comm"],
                latest_fit_metrics["download_comm"],
                latest_fit_metrics["total_comm"],
            ]
        )

    print()
    print("=" * 70)
    print(f"[Baseline Non-IID] Round {current_round}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Loss: {loss:.4f}")
    print(f"Total comm: {latest_fit_metrics['total_comm']:.2f} KB")
    print("=" * 70)
    print()

    return {
        "accuracy": accuracy,
        "loss": loss,
    }


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
)


print()
print("=" * 70)
print("FedAvg Baseline Non-IID Server")
print(f"Results: {RESULT_FILE}")
print("=" * 70)
print()


fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(
        num_rounds=NUM_ROUNDS,
    ),
    strategy=strategy,
)