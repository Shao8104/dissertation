# network_simulator.py

import random


# =========================================
# Network Simulator
# =========================================

def simulate_network_condition(
    client_id: int,
    server_round: int,
):
    """
    Simulate heterogeneous network conditions for each client
    in each federated learning round.

    Returns:
        bandwidth_mbps: simulated bandwidth in Mbps
        latency_ms: simulated latency in milliseconds
    """

    random.seed(
        client_id * 1000 + server_round
    )

    # Different clients have different baseline network quality
    if client_id == 0:
        # Good network client
        bandwidth_mbps = random.uniform(7.0, 10.0)
        latency_ms = random.uniform(10.0, 60.0)

    elif client_id == 1:
        # Medium-good network client
        bandwidth_mbps = random.uniform(4.0, 7.0)
        latency_ms = random.uniform(50.0, 120.0)

    elif client_id == 2:
        # Medium network client
        bandwidth_mbps = random.uniform(2.0, 5.0)
        latency_ms = random.uniform(100.0, 180.0)

    elif client_id == 3:
        # Poor network client
        bandwidth_mbps = random.uniform(0.8, 3.0)
        latency_ms = random.uniform(160.0, 260.0)

    else:
        # Highly unstable network client
        bandwidth_mbps = random.uniform(0.5, 10.0)
        latency_ms = random.uniform(20.0, 300.0)

    return (
        float(bandwidth_mbps),
        float(latency_ms),
    )


# =========================================
# Optional Utility Function
# =========================================

def estimate_transmission_time(
    payload_bytes: int,
    bandwidth_mbps: float,
    latency_ms: float,
):
    """
    Estimate transmission time under simulated network conditions.

    This is optional and can be used later for analysing
    communication-computation trade-off.

    Formula:
        transmission_time = payload_size / bandwidth + latency
    """

    payload_bits = payload_bytes * 8

    bandwidth_bps = bandwidth_mbps * 1_000_000

    transmission_time_seconds = (
        payload_bits / bandwidth_bps
    ) + (
        latency_ms / 1000
    )

    return float(transmission_time_seconds)


# =========================================
# Test
# =========================================

if __name__ == "__main__":

    for round_id in range(1, 6):

        print(f"\nRound {round_id}")

        for client_id in range(5):

            bandwidth, latency = simulate_network_condition(
                client_id=client_id,
                server_round=round_id,
            )

            print(
                f"Client {client_id} | "
                f"Bandwidth: {bandwidth:.2f} Mbps | "
                f"Latency: {latency:.2f} ms"
            )