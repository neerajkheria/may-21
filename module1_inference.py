import torch
import time
import json

# Configuration
BATCH_SIZE = 64
INPUT_DIM = 768
HIDDEN_DIM = 2048
OUTPUT_DIM = 256
NUM_RUNS = 100


class SupportQueryEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(INPUT_DIM, HIDDEN_DIM),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),

            torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            torch.nn.ReLU(),

            torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        )

    def forward(self, x):
        return self.layers(x)


def benchmark_device(device_name: str, model: torch.nn.Module):

    device = torch.device(device_name)

    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(BATCH_SIZE, INPUT_DIM).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    times = []

    with torch.no_grad():

        for _ in range(NUM_RUNS):

            start = time.perf_counter()

            output = model(dummy_input)

            if device_name == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            times.append((end - start) * 1000)

    avg_ms = sum(times) / len(times)

    throughput = (BATCH_SIZE * 1000) / avg_ms

    return {
        "device": device_name.upper(),
        "avg_latency_ms": round(avg_ms, 3),
        "min_latency_ms": round(min(times), 3),
        "max_latency_ms": round(max(times), 3),
        "throughput_qps": round(throughput, 1),
        "output_shape": list(output.shape)
    }


def main():

    model = SupportQueryEncoder()

    print("=" * 60)
    print("AI INFERENCE BENCHMARK")
    print("=" * 60)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    cpu_result = benchmark_device("cpu", model)

    print("\nCPU RESULTS")
    print(json.dumps(cpu_result, indent=2))

    if torch.cuda.is_available():

        gpu_result = benchmark_device("cuda", model)

        print("\nGPU RESULTS")
        print(json.dumps(gpu_result, indent=2))

        speedup = (
            cpu_result["avg_latency_ms"] /
            gpu_result["avg_latency_ms"]
        )

        print(f"\nGPU Speedup: {speedup:.2f}x")

    else:
        print("\nCUDA GPU not available")


if __name__ == "__main__":
    main()