# scripts/benchmark_latency.py
import random
import statistics
import time
import httpx

URL = "http://127.0.0.1:8000/predict"
N = 200

random.seed(42)
times_ms = []
with httpx.Client() as client:
    client.post(URL, json={"user_id": 1})  # warm-up, not counted
    for _ in range(N):
        user_id = random.randint(1, 2000)
        start = time.perf_counter()
        response = client.post(URL, json={"user_id": user_id})
        times_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200

q = statistics.quantiles(times_ms, n=100)
print(f"requests: {N}")
print(f"p50: {q[49]:.1f} ms")
print(f"p95: {q[94]:.1f} ms")
print(f"p99: {q[98]:.1f} ms")
