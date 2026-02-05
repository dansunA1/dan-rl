"""Benchmark tests for sampling and recv performance"""
import time
import statistics
import sys

print("[DEBUG] Starting imports...", file=sys.stderr, flush=True)
from danrl._test_dan_rl_python import RuleSamplingWorker, EnvConfig, ActorConfig
print("[DEBUG] Imports completed successfully", file=sys.stderr, flush=True)


def benchmark_sampling_recv(num_samples=1000, horizon=128, warmup_samples=10):
    """
    Benchmark the sampling and recv performance.
    
    Args:
        num_samples: Number of samples to receive for benchmarking
        horizon: Horizon size for sampling (affects batch size)
        warmup_samples: Number of samples to receive before starting timing
    
    Returns:
        dict with benchmark results including:
        - total_time: Total time taken
        - samples_per_second: Throughput
        - avg_recv_time: Average time per recv call
        - min_recv_time: Minimum recv time
        - max_recv_time: Maximum recv time
        - median_recv_time: Median recv time
    """
    # Create worker
    print("[DEBUG] Creating EnvConfig...", file=sys.stderr, flush=True)
    env_config = EnvConfig.MyEnv(x=1)
    print(f"[DEBUG] EnvConfig created: {env_config}", file=sys.stderr, flush=True)
    
    print("[DEBUG] Creating ActorConfig...", file=sys.stderr, flush=True)
    actor_config = ActorConfig.MyActorA(x=1)
    print(f"[DEBUG] ActorConfig created: {actor_config}", file=sys.stderr, flush=True)
    
    print("[DEBUG] Creating RuleSamplingWorker...", file=sys.stderr, flush=True)
    worker = RuleSamplingWorker(env_config, actor_config)
    print("[DEBUG] RuleSamplingWorker created successfully", file=sys.stderr, flush=True)
    
    # Start sampling in background
    print(f"[DEBUG] Calling worker.start(horizon={horizon})...", file=sys.stderr, flush=True)
    worker.start(horizon)
    print("[DEBUG] worker.start() completed successfully", file=sys.stderr, flush=True)
    
    # Warmup: receive a few samples to ensure everything is initialized
    print(f"Warming up with {warmup_samples} samples...")
    for i in range(warmup_samples):
        print(f"[DEBUG] Warmup recv attempt {i+1}/{warmup_samples}...", file=sys.stderr, flush=True)
        try:
            sampling = worker.recv()
            print(f"[DEBUG] Warmup recv {i+1} succeeded", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[DEBUG] Warmup recv {i+1} exception: {e}", file=sys.stderr, flush=True)
            if "Failed to receive" in str(e):
                time.sleep(0.1)  # Wait a bit if buffer is empty
                continue
            raise
    
    # Benchmark: measure recv times
    print(f"Benchmarking {num_samples} samples...")
    recv_times = []
    total_start = time.perf_counter()
    
    samples_received = 0
    while samples_received < num_samples:
        print(f"[DEBUG] Benchmark recv attempt {samples_received+1}/{num_samples}...", file=sys.stderr, flush=True)
        recv_start = time.perf_counter()
        try:
            sampling = worker.recv()
            recv_end = time.perf_counter()
            recv_times.append(recv_end - recv_start)
            samples_received += 1
            print(f"[DEBUG] Benchmark recv {samples_received} succeeded", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[DEBUG] Benchmark recv exception: {e}", file=sys.stderr, flush=True)
            if "Failed to receive" in str(e):
                # Buffer empty, wait a bit
                time.sleep(0.001)
                continue
            raise
    
    total_end = time.perf_counter()
    total_time = total_end - total_start
    
    # Calculate statistics
    if recv_times:
        avg_recv_time = statistics.mean(recv_times)
        min_recv_time = min(recv_times)
        max_recv_time = max(recv_times)
        median_recv_time = statistics.median(recv_times)
    else:
        avg_recv_time = min_recv_time = max_recv_time = median_recv_time = 0.0
    
    samples_per_second = num_samples / total_time if total_time > 0 else 0
    
    results = {
        'total_time': total_time,
        'samples_per_second': samples_per_second,
        'avg_recv_time': avg_recv_time,
        'min_recv_time': min_recv_time,
        'max_recv_time': max_recv_time,
        'median_recv_time': median_recv_time,
        'num_samples': num_samples,
        'horizon': horizon,
    }
    
    return results


def print_benchmark_results(results):
    """Print benchmark results in a readable format"""
    print("\n" + "="*60)
    print("SAMPLING/RECV BENCHMARK RESULTS")
    print("="*60)
    print(f"Configuration:")
    print(f"  Horizon: {results['horizon']}")
    print(f"  Samples: {results['num_samples']}")
    print(f"\nPerformance:")
    print(f"  Total time: {results['total_time']:.4f} seconds")
    print(f"  Samples/second: {results['samples_per_second']:.2f}")
    print(f"\nRecv timing (per call):")
    print(f"  Average: {results['avg_recv_time']*1000:.4f} ms")
    print(f"  Median:  {results['median_recv_time']*1000:.4f} ms")
    print(f"  Min:     {results['min_recv_time']*1000:.4f} ms")
    print(f"  Max:     {results['max_recv_time']*1000:.4f} ms")
    print("="*60 + "\n")


def run_benchmark_suite():
    """Run a suite of benchmarks with different configurations"""
    configs = [
        {'horizon': 256, 'num_samples': 1000},
        {'horizon': 512, 'num_samples': 1000},
        {'horizon': 1024, 'num_samples': 500},
        {'horizon': 2048, 'num_samples': 500},
        {'horizon': 4096, 'num_samples': 200},
    ]
    
    all_results = []
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Running benchmark: horizon={config['horizon']}, samples={config['num_samples']}")
        print(f"{'='*60}")
        results = benchmark_sampling_recv(
            num_samples=config['num_samples'],
            horizon=config['horizon'],
            warmup_samples=5
        )
        print_benchmark_results(results)
        all_results.append(results)
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Horizon':<10} {'Samples/s':<15} {'Avg Recv (ms)':<15}")
    print("-"*60)
    for r in all_results:
        print(f"{r['horizon']:<10} {r['samples_per_second']:<15.2f} {r['avg_recv_time']*1000:<15.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run quick benchmark by default
    print("[DEBUG] Entering main block", file=sys.stderr, flush=True)
    print("Running quick benchmark...")
    print("[DEBUG] Calling benchmark_sampling_recv...", file=sys.stderr, flush=True)
    results = benchmark_sampling_recv(num_samples=100, horizon=128, warmup_samples=5)
    print("[DEBUG] benchmark_sampling_recv completed, printing results...", file=sys.stderr, flush=True)
    print_benchmark_results(results)
    print("[DEBUG] Script completed successfully", file=sys.stderr, flush=True)
    
    # Uncomment to run full benchmark suite
    # run_benchmark_suite()

