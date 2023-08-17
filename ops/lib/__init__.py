import os
import sysconfig
import torch

_HERE = os.path.realpath(os.path.dirname(__file__))
EXT_SUFFIX = sysconfig.get_config_var('EXT_SUFFIX')

if (torch.cuda.is_available()):
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
