import os
import sysconfig
import torch

_HERE = os.path.realpath(os.path.dirname(__file__))
EXT_SUFFIX = sysconfig.get_config_var('EXT_SUFFIX')

torch.ops.load_library(_HERE + '/ops_cc.so')
print(dir(torch.ops.ops_cc))
if torch.cuda.is_available():
    torch.ops.load_library(_HERE + '/ops_cuda.so')
    print(dir(torch.ops.ops_cuda))
    exit()


def outer_product(X, Y, sender_list, nnodes):
    return torch.ops.ops_cu.outer_product(X, Y, sender_list, nnodes)


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
        result = torch.ops.ops_cu.ops_gpu(tensor_a, tensor_b, scatter_indices, out_dim)
    else:
        result = torch.ops.ops_cc.forward(tensor_a, tensor_b, scatter_indices, out_dim)

    return result
