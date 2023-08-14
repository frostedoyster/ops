import torch
from ops import ref_ops, opt_ops
import time


def benchmark(dtype, device):
    
    print(f"Benchmarking dtype {dtype} and device {device}")
    a = torch.rand((100000, 20), dtype=dtype, device=device, requires_grad=True)
    b = torch.rand((100000, 5), dtype=dtype, device=device, requires_grad=True)
    indices = torch.sort(torch.randint(1000, (100000,), device=device))[0]

    # Warm-up:
    loss = torch.sum(ref_ops(a, b, indices, 1000))
    loss.backward()
    loss = torch.sum(opt_ops(a, b, indices, 1000))
    loss.backward()

    start = time.time()
    for _ in range(1000):
        loss = torch.sum(ref_ops(a, b, indices, 1000))
        loss.backward()
    finish = time.time()
    print(f"The pure torch implementation took {finish-start} seconds")

    start = time.time()
    for _ in range(1000):
        loss = torch.sum(opt_ops(a, b, indices, 1000))
        loss.backward()
    finish = time.time()
    print(f"The optimized implementation took {finish-start} seconds")


if __name__ == "__main__":
    benchmark(torch.float32, "cpu")
    benchmark(torch.float64, "cpu")
    #if torch.cuda.is_available():
    #    benchmark(torch.float32, "cuda")
    #    benchmark(torch.float64, "cuda")
