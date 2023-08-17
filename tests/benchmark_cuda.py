import torch
from ops.lib.ops import ref_ops, opt_ops
import time
from ops import lib # forces torch.load_library


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
        neighbour_cuda = lib.calculate_neighbours(
            indices_cuda, nnodes, 64)
    finish = time.time()
    print(f"The indices for CUDA implementation took {finish-start:.3f} seconds")

    X = a.cuda()
    Y = b.cuda()

    # warmup
    for _ in range(1000):
        output_cuda = lib.forward(X, Y, neighbour_cuda, nnodes,  32, 4, 1)

    start = time.time()
    for _ in range(1000):
        output_cuda = lib.forward(X, Y, neighbour_cuda, nnodes,  32, 4, 1)
    torch.cuda.synchronize()
    finish = time.time()
    print(f"The CUDA implementation forward (direct call) took {finish-start:.3f} seconds")

    grad_in = torch.ones_like(output_cuda)

    start = time.time()
    for _ in range(1000):
        dX, dY = lib.backward(
            X, Y, grad_in, neighbour_cuda, nnodes)
    torch.cuda.synchronize()
    finish = time.time()
    print(
        f"The CUDA implementation backward (direct call) took {finish-start:.3f} seconds")

    start = time.time()
    for _ in range(1000):
        v = opt_ops(X, Y, indices_cuda, nnodes)
    torch.cuda.synchronize()
    finish = time.time()
    print(f"The CUDA implementation forward (python autograd) took {finish-start:.3f} seconds")

    start = time.time()
    for _ in range(1000):
        loss = torch.sum(opt_ops(X, Y, indices_cuda, nnodes))
        loss.backward()
    torch.cuda.synchronize()
    finish = time.time()
    print(f"The CUDA implementation backward (python autograd) took {finish-start:.3f} seconds")

    start = time.time()
    for _ in range(1000):
        v = lib.outer_product(X, Y, indices_cuda, nnodes)
    torch.cuda.synchronize()
    finish = time.time()
    print(f"The CUDA implementation forward (C++ autograd) took {finish-start:.3f} seconds")

    start = time.time()
    for _ in range(1000):
        loss = torch.sum(lib.outer_product(X, Y, indices_cuda, nnodes))
        loss.backward()
    torch.cuda.synchronize()
    finish = time.time()
    print(f"The CUDA implementation backward (C++ autograd) took {finish-start:.3f} seconds")

    start = time.time()
    for _ in range(1000):
        ref_ops(X, Y, indices_cuda, nnodes)
    torch.cuda.synchronize()
    finish = time.time()
    print(
        f"The ref torch implementation forward took {finish-start:.3f} seconds")

    start = time.time()
    for _ in range(1000):
        loss = torch.sum(ref_ops(X, Y, indices_cuda, nnodes))
        loss.backward()
    torch.cuda.synchronize()
    finish = time.time()
    print(f"The ref torch implementation backward took {finish-start:.3f} seconds")

    start = time.time()
    for _ in range(1000):
        output_cpu = opt_ops(a, b, indices, nnodes)
    finish = time.time()
    print(f"The CPU implementation forward took {finish-start:.3f} seconds")

    start = time.time()
    for _ in range(1000):
        loss = torch.sum(opt_ops(a, b, indices, nnodes))
        loss.backward()
    finish = time.time()
    print(f"The CPU implementation backward took {finish-start:.3f} seconds")


if __name__ == "__main__":
    benchmark(torch.float32, "cpu")
    # benchmark(torch.float64, "cpu")
    # if torch.cuda.is_available():
    #    benchmark(torch.float32, "cuda")
    #    benchmark(torch.float64, "cuda")
