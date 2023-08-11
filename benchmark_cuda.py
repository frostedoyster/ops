import torch
from ops import ref_ops, opt_ops
import time
import ops_cuda


def benchmark(dtype, device):

    nedges = 60000
    nnodes = 1000
    nfeatures = 32
    nl = 5

    print(f"--DTYPE: {dtype}")
    print(f"Benchmarking dtype {dtype} and device {device}")
    print(f"nodes: {nnodes} and edges: {nedges}")
    a = torch.rand((nedges, nfeatures), dtype=dtype,
                   device=device, requires_grad=True)
    b = torch.rand((nedges, nl), dtype=dtype,
                   device=device, requires_grad=True)
    indices = torch.sort(torch.randint(nnodes, (nedges,), device=device))[0]

    indices_cuda = indices.cuda().int()

    start = time.time()
    for _ in range(100):
        neighbour_cuda = ops_cuda.calculate_neighbours(
            indices_cuda, nnodes, 64)
    finish = time.time()
    print(f"The indices for CUDA implementation took {finish-start} seconds")

    X = a.cuda()
    Y = b.cuda()
    start = time.time()
    for _ in range(100):
        output_cuda = ops_cuda.forward(X, Y, neighbour_cuda, nnodes,  32, 4, 1)
    finish = time.time()
    print(f"The CUDA implementation 1 forward took {finish-start} seconds")

    start = time.time()
    for _ in range(100):
        output_cpu = opt_ops(a, b, indices, nnodes)
    finish = time.time()
    print(f"The CPU implementation forward took {finish-start} seconds")

    start = time.time()
    for _ in range(100):
        loss = torch.sum(opt_ops(a, b, indices, nnodes))
        loss.backward()
    finish = time.time()
    print(f"The CPU implementation backward took {finish-start} seconds")


if __name__ == "__main__":
    benchmark(torch.float32, "cpu")
    benchmark(torch.float64, "cpu")
    # if torch.cuda.is_available():
    #    benchmark(torch.float32, "cuda")
    #    benchmark(torch.float64, "cuda")
