import os
import sysconfig
import torch


import os
import sysconfig
import torch

_HERE = os.path.realpath(os.path.dirname(__file__))
EXT_SUFFIX = sysconfig.get_config_var('EXT_SUFFIX')

torch.ops.load_library(_HERE + '/ops_cc.so')
if torch.cuda.is_available():
    torch.ops.load_library(_HERE + '/ops_cuda.so')

'''
    differenciable outer product
'''

def outer_product(X, Y, sender_list, nnodes):
    return torch.ops.ops.outer_product(X, Y, sender_list, nnodes)

'''
    Non-differenciable - for testing only.
'''


def forward(X, Y, sender_list, nnodes, nthreadx=32, nthready=4, nthreadz=1):
    return torch.ops.ops.forward(X, Y, sender_list, nnodes, nthreadx, nthready, nthreadz)


def backward(X, Y, grad_in, neighbours, nnodes):
    return torch.ops.ops.backward(X, Y, grad_in, neighbours, nnodes)


def calculate_neighbours(sender_list, nnodes, nthreadx=64):
    return torch.ops.ops.calculate_neighbours(sender_list, nnodes, nthreadx)


def ref_ops(tensor_a, tensor_b, scatter_indices, out_dim):
    # Reference implementation

    assert tensor_a.shape[0] == tensor_b.shape[0]
    if torch.max(scatter_indices) > out_dim:
        raise ValueError(
            "The highest scatter index is greater than the output dimension")

    result = torch.zeros(
        (out_dim, tensor_a.shape[1], tensor_b.shape[1]), dtype=tensor_a.dtype, device=tensor_b.device)
    result.index_add_(
        dim=0,
        index=scatter_indices,
        source=tensor_a.unsqueeze(2)*tensor_b.unsqueeze(1)
    )
    return result


def opt_ops(tensor_a, tensor_b, scatter_indices, out_dim):

    if tensor_a.device != tensor_b.device:
        raise ValueError("All tensors must be on the same device")
    if tensor_a.device != scatter_indices.device:
        raise ValueError("All tensors must be on the same device")
    if tensor_a.dtype != tensor_b.dtype:
        raise ValueError("The two float tensors must have the same dtype")
    
    tensor_a = tensor_a.contiguous()
    tensor_b = tensor_b.contiguous()
    scatter_indices = scatter_indices.contiguous()

    if tensor_a.is_cuda:
        result = torch.ops.ops_cu.ops(tensor_a, tensor_b, scatter_indices.to(torch.int32), out_dim).swapaxes(1, 2)
    else:
        result = torch.ops.ops_cc.ops(tensor_a, tensor_b, scatter_indices, out_dim)

    return result

