import torch
import ops_cc

HAS_CUDA = False
if torch.cuda.is_available():
    import ops_cuda
    HAS_CUDA = True


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


class OptOps(torch.autograd.Function):
    # Optimized implementation

    @staticmethod
    def forward(ctx, tensor_a, tensor_b, scatter_indices, out_dim):

        if tensor_a.device != tensor_b.device:
            raise ValueError("All tensors must be on the same device")
        if tensor_a.device != scatter_indices.device:
            raise ValueError("All tensors must be on the same device")
        if tensor_a.dtype != tensor_b.dtype:
            raise ValueError("The two float tensors must have the same dtype")

        # don't think these are needed.
        # tensor_a = tensor_a.contiguous()
        # tensor_b = tensor_b.contiguous()
        # scatter_indices = scatter_indices.contiguous()

        if tensor_a.is_cuda:
            first_occurrences = ops_cuda.calculate_neighbours(
                scatter_indices.int(), out_dim, 64)
        else:
            first_occurrences = ops_cc.find_first_occurrences(
                scatter_indices, out_dim)

        ctx.out_dim = out_dim

        if (tensor_a.requires_grad or tensor_b.requires_grad):
            ctx.save_for_backward(tensor_a, tensor_b,
                                  scatter_indices, first_occurrences)

        if tensor_a.is_cuda:
            # transpose to make format similar to C code.
            return ops_cuda.forward(tensor_a, tensor_b, first_occurrences, out_dim,  32, 4, 1).transpose(-1, -2)
        else:
            return ops_cc.forward(tensor_a, tensor_b, scatter_indices, first_occurrences, out_dim)

    @staticmethod
    def backward(ctx, grad_output):

        tensor_a, tensor_b, scatter_indices, first_occurrences = ctx.saved_variables

        if grad_output.is_cuda:

            out_dim = ctx.out_dim
            result = ops_cuda.backward(
                tensor_a, tensor_b, grad_output.transpose(-1, -2), first_occurrences, out_dim, 128, 1, 1)  # convert grad_output to CUDA ordering.
        else:
            out_dim = ctx.out_dim
            result = ops_cc.backward(
                grad_output, tensor_a, tensor_b, scatter_indices, first_occurrences, out_dim)

        return result[0], result[1], None, None


opt_ops = OptOps.apply  # simply rename the function to make it easier to call