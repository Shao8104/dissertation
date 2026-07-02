
# adaptive_policy.py

# =========================================
# Lightweight Training-aware
# Network-aware Adaptive Quantization
# =========================================


class AdaptivePolicy:

    def __init__(self):

        # Save previous loss

        self.prev_loss = None

    # =====================================
    # Normalize bandwidth
    # =====================================

    def normalize_bandwidth(

        self,

        bandwidth_mbps,

    ):

        return min(

            bandwidth_mbps / 10.0,

            1.0

        )

    # =====================================
    # Normalize latency
    # =====================================

    def normalize_latency(

        self,

        latency_ms,

    ):

        return 1.0 - min(

            latency_ms / 300.0,

            1.0

        )

    # =====================================
    # Network Quality Score
    # =====================================

    def calculate_network_score(

        self,

        bandwidth_mbps,

        latency_ms,

    ):

        bw_score = self.normalize_bandwidth(

            bandwidth_mbps

        )

        lat_score = self.normalize_latency(

            latency_ms

        )

        network_score = (

            0.7 * bw_score

            +

            0.3 * lat_score

        )

        return network_score

    # =====================================
    # Loss Trend
    # =====================================

    def calculate_loss_change(

        self,

        current_loss,

    ):

        if self.prev_loss is None:

            self.prev_loss = current_loss

            return 1.0

        delta = (

            self.prev_loss

            -

            current_loss

        )

        self.prev_loss = current_loss

        return delta

    # =====================================
    # Choose Bit-width
    # =====================================

    def choose_bitwidth(

        self,

        bandwidth_mbps,

        latency_ms,

        train_loss,

        server_round,

    ):

        network_score = (

            self.calculate_network_score(

                bandwidth_mbps,

                latency_ms,

            )

        )

        loss_change = (

            self.calculate_loss_change(

                train_loss

            )

        )

        # -------------------------------

        # Stage 1

        # Early training

        # Use higher precision

        # -------------------------------

        if server_round <= 2:

            bit_width = 8

        # -------------------------------

        # Stage 2

        # Loss decreases slowly

        # protect convergence

        # -------------------------------

        elif loss_change < 0.01:

            bit_width = 8

        # -------------------------------

        # Stage 3

        # Adaptive decision

        # -------------------------------

        else:

            if network_score < 0.35:

                bit_width = 4

            elif network_score < 0.70:

                bit_width = 6

            else:

                bit_width = 8

        return {

            "bit_width":

                int(bit_width),

            "network_score":

                float(network_score),

            "loss_change":

                float(loss_change),

        }


# =========================================
# Test
# =========================================

if __name__ == "__main__":

    policy = AdaptivePolicy()

    examples = [

        # round

        # bandwidth

        # latency

        # loss

        (1, 8, 20, 1.2),

        (2, 6, 50, 0.8),

        (3, 3, 100, 0.5),

        (4, 1, 220, 0.3),

        (5, 0.8, 260, 0.15),

        (6, 8, 30, 0.14),

    ]

    for r, bw, lat, loss in examples:

        result = policy.choose_bitwidth(

            bandwidth_mbps=bw,

            latency_ms=lat,

            train_loss=loss,

            server_round=r,

        )

        print()

        print(

            f"Round: {r}"

        )

        print(

            f"Bandwidth: "

            f"{bw:.2f} Mbps"

        )

        print(

            f"Latency: "

            f"{lat:.2f} ms"

        )

        print(

            f"Train Loss: "

            f"{loss:.4f}"

        )

        print(

            f"Network Score: "

            f"{result['network_score']:.4f}"

        )

        print(

            f"Loss Change: "

            f"{result['loss_change']:.4f}"

        )

        print(

            f"Selected Bit-width: "

            f"{result['bit_width']}"

        )
