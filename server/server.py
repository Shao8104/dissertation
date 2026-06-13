# server/server.py

import flwr as fl

strategy = fl.server.strategy.FedAvg(

    min_fit_clients=5,

    min_evaluate_clients=5,

    min_available_clients=5,

)

fl.server.start_server(

    server_address="0.0.0.0:8080",

    config=fl.server.ServerConfig(
        num_rounds=10
    ),

    strategy=strategy,

)