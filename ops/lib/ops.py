import torch


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
        result = torch.ops.ops_cu.ops_gpu(tensor_a, tensor_b, scatter_indices, out_dim)
    else:
        result = torch.ops.ops_cc.forward(tensor_a, tensor_b, scatter_indices, out_dim)

    return result

