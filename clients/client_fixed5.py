import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

from clients.flower_client_fixed import FlowerClient

import flwr as fl


fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(4),
)