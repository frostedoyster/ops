import torch
from ops.lib.ops import ref_ops, opt_ops
import time


def benchmark(dtype, device):

    nnodes = 1000
    nedges = 60000
    nchannels = 32
    nl = 5

    print(f"Benchmarking dtype {dtype} and device {device}")
    a = torch.rand((nedges, nchannels), dtype=dtype,
                   device=device, requires_grad=True)
    b = torch.rand((nedges, nl), dtype=dtype,
                   device=device, requires_grad=True)
    indices = torch.sort(torch.randint(nnodes, (nedges,), device=device))[0]

    # Warm-up:
    loss = torch.sum(ref_ops(a, b, indices, nnodes))
    loss.backward()
    loss = torch.sum(opt_ops(a, b, indices, nnodes))
    loss.backward()

    start = time.time()
    for _ in range(1000):
        ref_ops(a, b, indices, nnodes)
    torch.cuda.synchronize()
    finish = time.time()
    print(f"The pure torch implementation fwd took {finish-start} seconds")

    start = time.time()
    for _ in range(1000):
        loss = torch.sum(ref_ops(a, b, indices, nnodes))
        loss.backward()
    torch.cuda.synchronize()
    finish = time.time()
    print(
        f"The pure torch implementation fwd + bwd took {finish-start} seconds")

    start = time.time()
    for _ in range(1000):
        opt_ops(a, b, indices, nnodes)
    torch.cuda.synchronize()
    finish = time.time()
    print(f"The optimized implementation fwd took {finish-start} seconds")

    start = time.time()
    for _ in range(1000):
        loss = torch.sum(opt_ops(a, b, indices, nnodes))
        loss.backward()
    torch.cuda.synchronize()
    finish = time.time()
    print(
        f"The optimized implementation fwd + bwd took {finish-start} seconds")


if __name__ == "__main__":
    # benchmark(torch.float32, "cpu")
    # benchmark(torch.float64, "cpu")
    if torch.cuda.is_available():
        benchmark(torch.float32, "cuda")
        benchmark(torch.float64, "cuda")
