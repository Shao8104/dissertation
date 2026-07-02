import flwr as fl

from flower_client_adaptive import FlowerClient


print()

print("=" * 70)

print("Adaptive Quantization Client 0")

print("=" * 70)

print()


client = FlowerClient(

    client_id=0

)


fl.client.start_numpy_client(

    server_address="127.0.0.1:8080",

    client=client,

)