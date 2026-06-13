import flwr as fl

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from clients.client import client_fn


strategy = fl.server.strategy.FedAvg()

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(
        num_rounds=3
    ),
    strategy=strategy,
)