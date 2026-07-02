# adaptive_policy.py

# =========================================
# Lightweight Training-aware
# Network-aware Adaptive Quantization
# =========================================
import numpy as np



class AdaptivePolicy:

    def __init__(self):
        # Save previous loss
        self.prev_loss = None

    # =====================================
    # Normalize bandwidth
    # =====================================
    def normalize_bandwidth(self, bandwidth_mbps):
        return min(bandwidth_mbps / 10.0, 1.0)

    # =====================================
    # Normalize latency
    # =====================================
    def normalize_latency(self, latency_ms):
        return 1.0 - min(latency_ms / 300.0, 1.0)

    # =====================================
    # Network Quality Score
    # =====================================
    def calculate_network_score(self, bandwidth_mbps, latency_ms):

        bw_score = self.normalize_bandwidth(bandwidth_mbps)
        lat_score = self.normalize_latency(latency_ms)

        network_score = (
            0.7 * bw_score +
            0.3 * lat_score
        )

        return network_score

    # =====================================
    # Loss Trend (kept for reference)
    # =====================================
    def calculate_loss_change(self, current_loss):

        if self.prev_loss is None:
            self.prev_loss = current_loss
            return 1.0

        delta = abs(self.prev_loss - current_loss)
        self.prev_loss = current_loss

        return delta

    # =====================================
    # Choose Bit-width (UPDATED VERSION)
    # =====================================
    def choose_bitwidth(
        self,
        bandwidth_mbps,
        latency_ms,
        train_loss,
        server_round,
    ):

        # -----------------------------
        # Network score
        # -----------------------------
        network_score = self.calculate_network_score(
            bandwidth_mbps,
            latency_ms,
        )

        # -----------------------------
        # Training dynamics
        # -----------------------------
        loss_change = self.calculate_loss_change(train_loss)

        # Smooth training signal (IMPORTANT FIX)

        training_score = min(1.0, loss_change * 8.0)

        # -----------------------------
        # Fusion score
        # -----------------------------
        final_score = (
            0.6 * network_score +
            0.4 * training_score
        )

        # -----------------------------
        # Bit-width decision
        # -----------------------------

        # Warm-up phase
        if server_round <= 2:
            bit_width = 8

        # Strong convergence phase (still allow adaptation)
        elif final_score < 0.20:
            bit_width = 4

        elif final_score < 0.60:
            bit_width = 6

        else:
            bit_width = 8

        return {
            "bit_width": int(bit_width),
            "network_score": float(network_score),
            "loss_change": float(loss_change),
            "training_score": float(training_score),
            "final_score": float(final_score),
        }


# =========================================
# Test
# =========================================
if __name__ == "__main__":

    policy = AdaptivePolicy()

    examples = [
        # round, bandwidth, latency, loss
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

        print("\n" + "=" * 60)
        print(f"Round: {r}")
        print(f"Bandwidth: {bw:.2f} Mbps")
        print(f"Latency: {lat:.2f} ms")
        print(f"Train Loss: {loss:.4f}")
        print(f"Network Score: {result['network_score']:.4f}")
        print(f"Training Score: {result['training_score']:.4f}")
        print(f"Final Score: {result['final_score']:.4f}")
        print(f"Loss Change: {result['loss_change']:.4f}")
        print(f"Selected Bit-width: {result['bit_width']}")