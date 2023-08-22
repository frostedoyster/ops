import torch
from ops.lib.ops import ref_ops, opt_ops
import time
from ops.lib import ops



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

    a_copy = a.clone().detach().requires_grad_(True)
    b_copy = b.clone().detach().requires_grad_(True)

    indices = torch.sort(torch.randint(nnodes, (nedges,), device=device))[0]

    first_occurrences_cpu = torch.ops.ops_cc.find_first_occurrences(
        indices.cpu(), nnodes)

    first_occurrences_cuda = ops.calculate_neighbours(
        indices.int(), nnodes, 64)

    print("first occurences consistent? ", torch.allclose(
        first_occurrences_cpu, first_occurrences_cuda.long().cpu()))

    ref_output = ref_ops(a, b, indices, nnodes)
    cuda_output = opt_ops(a_copy, b_copy, indices, nnodes)

    print("forwards consistent? ", torch.allclose(ref_output, cuda_output))

    loss = torch.sum(ref_output)
    loss.backward()

    loss = torch.sum(cuda_output)
    loss.backward()

    print("bwd consistent wrt. a? ", torch.allclose(a.grad, a_copy.grad))
    print("bwd consistent wrt. b? ", torch.allclose(b.grad, b_copy.grad))


if __name__ == "__main__":
    if torch.cuda.is_available():
        benchmark(torch.float32, "cuda")
