import torch
from ops import ref_ops, opt_ops
import time
import ops_cuda


def benchmark(dtype, device):

    nedges = 30000
    nnodes = 1000
    nfeatures = 96
    nl = 16

    print(f"--DTYPE: {dtype}")
    print(f"Benchmarking dtype {dtype} and device {device}")
    print(f"nodes: {nnodes} and edges: {nedges}")
    print(f"nfeatures: {nfeatures} and nsphericalharmonics: {nl}")
    a = torch.rand((nedges, nfeatures), dtype=dtype,
                   device=device, requires_grad=True)
    b = torch.rand((nedges, nl), dtype=dtype,
                   device=device, requires_grad=True)
    indices = torch.sort(torch.randint(nnodes, (nedges,), device=device))[0]

    indices_cuda = indices.cuda().int()

    start = time.time()
    for _ in range(1000):
        neighbour_cuda = ops_cuda.calculate_neighbours(
            indices_cuda, nnodes, 64)
    finish = time.time()
    print(f"The indices for CUDA implementation took {finish-start} seconds")

    X = a.cuda()
    Y = b.cuda()

    #warmup
    for _ in range(1000):
        output_cuda = ops_cuda.forward(X, Y, neighbour_cuda, nnodes,  32, 4, 1)

    start = time.time()
    for _ in range(1000):
        output_cuda = ops_cuda.forward(X, Y, neighbour_cuda, nnodes,  32, 4, 1)
    finish = time.time()
    print(f"The CUDA implementation 1 forward took {finish-start} seconds")

    start = time.time()
    for _ in range(1000):
        output_cuda = ops_cuda.forward2(X, Y, neighbour_cuda, nnodes,  32, 4, 1)
    finish = time.time()
    print(f"The CUDA implementation 2 forward took {finish-start} seconds")

    start = time.time()
    for _ in range(1000):
        output_cuda = ops_cuda.forward3(X, Y, neighbour_cuda, nnodes,  32, 4, 1)
    finish = time.time()
    print(f"The CUDA implementation 3 forward took {finish-start} seconds")

    print (output_cuda[0])

    start = time.time()
    for _ in range(1000):
        output_cpu = opt_ops(a, b, indices, nnodes)
    finish = time.time()
    print(f"The CPU implementation forward took {finish-start} seconds")

    start = time.time()
    for _ in range(1000):
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
