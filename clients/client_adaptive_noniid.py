import sys
import flwr as fl

from flower_client_adaptive_noniid import FlowerClient


if len(sys.argv) != 2:
    print("Usage: python clients/client_adaptive_noniid.py <client_id>")
    sys.exit(1)

client_id = int(sys.argv[1])

client = FlowerClient(client_id)

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=client,
)