# server/server_baseline.py

import os
import csv

import flwr as fl


RESULT_DIR = "results"

RESULT_FILE = os.path.join(
    RESULT_DIR,
    "baseline_results.csv"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# =========================================
# Save CSV Header
# =========================================

with open(
    RESULT_FILE,
    mode="w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([

        "Round",

        "Accuracy",

        "Loss",

        "Train Loss",

        "Train Time(s)",

        "Upload Comm(KB)",

        "Download Comm(KB)",

        "Total Comm(KB)"

    ])


# =========================================
# Global Variables
# =========================================

current_round = 0

latest_fit_metrics = {

    "train_loss": 0,

    "train_time": 0,

    "upload_comm": 0,

    "download_comm": 0,

    "total_comm": 0,

}


# =========================================
# Aggregate Fit Metrics
# =========================================

def fit_metrics_aggregation_fn(metrics):

    global latest_fit_metrics

    total_examples = sum(

        num_examples

        for num_examples, _ in metrics

    )

    train_loss = sum(

        num_examples

        *

        m["train_loss"]

        for num_examples, m in metrics

    ) / total_examples


    train_time = sum(

        num_examples

        *

        m["train_time"]

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


# =========================================
# Aggregate Evaluate Metrics
# =========================================

def evaluate_metrics_aggregation_fn(metrics):

    global current_round

    global latest_fit_metrics

    current_round += 1


    total_examples = sum(

        num_examples

        for num_examples, _ in metrics

    )


    accuracy = sum(

        num_examples

        *

        m["accuracy"]

        for num_examples, m in metrics

    ) / total_examples


    loss = sum(

        num_examples

        *

        m["test_loss"]

        for num_examples, m in metrics

    ) / total_examples


    print()

    print("=" * 70)

    print(

        f"Round {current_round}"

    )

    print(

        f"Global Accuracy: "

        f"{accuracy:.4f}"

    )

    print(

        f"Global Loss: "

        f"{loss:.4f}"

    )

    print(

        f"Train Loss: "

        f"{latest_fit_metrics['train_loss']:.4f}"

    )

    print(

        f"Train Time: "

        f"{latest_fit_metrics['train_time']:.2f}s"

    )

    print(

        f"Total Comm: "

        f"{latest_fit_metrics['total_comm']:.2f} KB"

    )

    print("=" * 70)

    print()


    # Save CSV

    with open(

        RESULT_FILE,

        mode="a",

        newline=""

    ) as f:

        writer = csv.writer(f)

        writer.writerow([

            current_round,

            accuracy,

            loss,

            latest_fit_metrics["train_loss"],

            latest_fit_metrics["train_time"],

            latest_fit_metrics["upload_comm"],

            latest_fit_metrics["download_comm"],

            latest_fit_metrics["total_comm"]

        ])


    return {

        "accuracy": accuracy,

        "loss": loss,

    }


# =========================================
# FedAvg Strategy
# =========================================

strategy = fl.server.strategy.FedAvg(

    fraction_fit=1.0,

    fraction_evaluate=1.0,

    min_fit_clients=5,

    min_evaluate_clients=5,

    min_available_clients=5,

    fit_metrics_aggregation_fn=

    fit_metrics_aggregation_fn,

    evaluate_metrics_aggregation_fn=

    evaluate_metrics_aggregation_fn,

)


# =========================================
# Start Server
# =========================================

print()

print("=" * 70)

print(

    "FedAvg Baseline Server"

)

print(

    "Dataset: MNIST"

)

print(

    "Clients: 5"

)

print(

    "Rounds: 10"

)

print(

    "Communication: Float32"

)

print(

    f"Results: "

    f"{RESULT_FILE}"

)

print("=" * 70)

print()


fl.server.start_server(

    server_address="0.0.0.0:8080",

    config=fl.server.ServerConfig(

        num_rounds=10

    ),

    strategy=strategy,

)

