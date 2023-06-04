import flwr as fl
from client import main


if __name__ == "__main__":
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
    )

    #Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
    # fl.simulation.start_simulation(
    #     client_fn=main,
    #     num_clients=2,
    #     config=fl.server.ServerConfig(num_rounds=5),
    #     strategy=strategy,
    #     #client_resources=client_resources,
    # )
