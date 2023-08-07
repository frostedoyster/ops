import torch
import ops_cc

if torch.cuda.is_available():
    # import ops_cuda
    pass


def ref_ops(tensor_a, tensor_b, scatter_indices, out_dim):
    # Reference implementation

    assert tensor_a.shape[0] == tensor_b.shape[0]
    if torch.max(scatter_indices) > out_dim: raise ValueError("The highest scatter index is greater than the output dimension")

    result = torch.zeros((out_dim, tensor_a.shape[1], tensor_b.shape[1]), dtype=tensor_a.dtype, device=tensor_b.device)
    result.index_add_(
        dim=0,
        index=scatter_indices,
        source=tensor_a.unsqueeze(2)*tensor_b.unsqueeze(1)
    )
    return result


class OptOps(torch.autograd.Function):
    # Optimized implementation

    @staticmethod
    def forward(ctx, tensor_a, tensor_b, scatter_indices, out_dim):

        if tensor_a.device != tensor_b.device: raise ValueError("All tensors must be on the same device")
        if tensor_a.device != scatter_indices.device: raise ValueError("All tensors must be on the same device")
        if tensor_a.dtype != tensor_b.dtype: raise ValueError("The two float tensors must have the same dtype")
        
        ctx.save_for_backward(tensor_a, tensor_b, scatter_indices)

        if tensor_a.is_cuda:
            return ops_cuda.forward(tensor_a, tensor_b, scatter_indices, out_dim)
        else:
            return ops_cc.forward(tensor_a, tensor_b, scatter_indices, out_dim)

    @staticmethod
    def backward(ctx, grad_output):

        tensor_a, tensor_b, scatter_indices = ctx.saved_tensors
        out_dim = grad_output.shape[0]        
        
        if tensor_a.is_cuda:
            return ops_cuda.backward(grad_output, tensor_a, tensor_b, scatter_indices, out_dim)
        else:
            return ops_cc.backward(grad_output, tensor_a, tensor_b, scatter_indices, out_dim)

opt_ops = OptOps.apply  # simply rename the function to make it easier to call
