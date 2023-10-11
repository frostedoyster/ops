#include <torch/script.h>
#include <iostream>

#include "cuda_base.cuh"

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

#define FULL_MASK 0xffffffff

torch::Tensor forward_gpu(torch::Tensor X,
                          torch::Tensor Y,
                          torch::Tensor receiver_list,
                          torch::Tensor neighbour_indices,
                          int64_t natoms,
                          int64_t nthreadx,
                          int64_t nthready,
                          int64_t nthreadz)
{

    torch::Tensor output = torch::empty({natoms, Y.size(1), X.size(1)},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    int32_t nby = find_integer_divisor(X.size(1), nthreadx);

    dim3 block_dim(natoms, nby);

    dim3 grid_dim(nthreadx, nthready, 1);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "forward_gpu", ([&]
                                  {
                    size_t total_buff_size = 0;

                    //total_buff_size += nthreadx * Y.size(1) * sizeof(scalar_t);

                    forward_kernel<scalar_t><<<block_dim, grid_dim, total_buff_size>>>(
                        X.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                        Y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                        receiver_list.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                        neighbour_indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                        output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()); }));

    cudaDeviceSynchronize();

    return output;
}

std::vector<torch::Tensor> backward_gpu(torch::Tensor X,
                                        torch::Tensor Y,
                                        torch::Tensor grad_in,
                                        torch::Tensor receiver_list,
                                        torch::Tensor neighbour_indices,
                                        int64_t natoms)
{

    torch::Tensor gradX;

    if (X.requires_grad())
    {
        gradX = torch::empty_like(X,
                                  torch::TensorOptions()
                                      .dtype(X.dtype())
                                      .device(X.device()));
    }
    else
    {
        gradX = torch::Tensor();
    }

    torch::Tensor gradY;

    if (Y.requires_grad())
    {
        gradY = torch::empty_like(Y,
                                  torch::TensorOptions()
                                      .dtype(Y.dtype())
                                      .device(Y.device()));
    }
    else
    {
        gradX = torch::Tensor();
    }

    dim3 grid_dim_x(M_PER_BLOCK_X, FEATS_PER_BLOCK_Y, 1);
    int32_t nbx = find_integer_divisor(X.size(1), FEATS_PER_BLOCK_Y);
    dim3 block_dim_x(natoms, nbx);

    dim3 grid_dim_y(FEATS_PER_BLOCK_Y, M_PER_BLOCK_X, 1);
    int32_t nby = find_integer_divisor(Y.size(1), M_PER_BLOCK_X);
    dim3 block_dim_y(natoms, nby);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "backward_gpu",
        ([&]
         {
            if (X.requires_grad())
            {
                size_t buff_size_x = 0;
                buff_size_x += Y.size(1) * FEATS_PER_BLOCK_Y * sizeof(scalar_t);

                backward_dX_kernel<scalar_t><<<block_dim_x, grid_dim_x, buff_size_x>>>(
                    Y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    grad_in.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    receiver_list.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                    neighbour_indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                    gradX.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
            }

            if (Y.requires_grad())
            {
                size_t buff_size_y = 0;
                buff_size_y += X.size(1) * M_PER_BLOCK_X * sizeof(scalar_t);

                backward_dY_kernel<scalar_t><<<block_dim_y, grid_dim_y, buff_size_y>>>(
                    X.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    grad_in.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    receiver_list.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                    neighbour_indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                    gradY.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
            } }));

    cudaDeviceSynchronize();

    return {gradX, gradY};
}

torch::Tensor calculate_neighbours_gpu(torch::Tensor sender_list, int64_t natoms, int64_t nthreadx)
{
    torch::Tensor output_indices = torch::empty(natoms,
                                                torch::TensorOptions()
                                                    .dtype(sender_list.dtype())
                                                    .device(sender_list.device()));

    int32_t nbx = find_integer_divisor(sender_list.size(0), NEIGHBOUR_NEDGES_PER_BLOCK);

    dim3 block_dim(nbx);

    dim3 grid_dim(nthreadx, 1, 1);

    size_t total_buff_size = 0;

    total_buff_size += (NEIGHBOUR_NEDGES_PER_BLOCK + 1) * sizeof(int32_t);

    calculate_neighbours_kernel<<<block_dim, grid_dim, total_buff_size>>>(

        sender_list.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        output_indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>());

    cudaDeviceSynchronize();

    return output_indices;
}

/*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("calculate_neighbours", &calculate_neighbours_gpu, "computes neighbourlist starts from sender list.");
    m.def("forward", &forward_gpu, "ops forward GPU.");
    m.def("backward", &backward_gpu, "ops backward GPU.");
}
*/

class OuterProductAutograd : public Function<OuterProductAutograd>
{
public:
    static torch::Tensor forward(
        AutogradContext *ctx,
        torch::Tensor X,
        torch::Tensor Y,
        torch::Tensor receiver_list,
        int64_t natoms)
    {
        torch::Tensor neighbours = calculate_neighbours_gpu(receiver_list, natoms, 64);

        if (X.requires_grad() || Y.requires_grad())
        {
            ctx->saved_data["natoms"] = natoms;

            ctx->save_for_backward({X, Y, receiver_list, neighbours});
        }

        torch::Tensor result = forward_gpu(X, Y, receiver_list, neighbours, natoms, 32, 4, 1);

        return result;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs)
    {
        auto saved_variables = ctx->get_saved_variables();

        auto X = saved_variables[0];
        auto Y = saved_variables[1];
        auto receiver_list = saved_variables[2];
        auto neighbours = saved_variables[3];

        int64_t natoms = ctx->saved_data["natoms"].toInt();

        // cout << "grad_outputs shape: " << grad_outputs[0].sizes() <<  endl;

        auto result = backward_gpu(X, Y, grad_outputs[0], receiver_list, neighbours, natoms);

        torch::Tensor undef;

        return {result[0], result[1], undef, undef};
    }
};

torch::Tensor ops(torch::Tensor X, torch::Tensor Y, torch::Tensor sender_list, int64_t natoms)
{
    return OuterProductAutograd::apply(X, Y, sender_list, natoms);
}

TORCH_LIBRARY(ops_cu, m)
{
    m.def("ops", &ops);
    m.def("calculate_neighbours", &calculate_neighbours_gpu);
    m.def("forward", &forward_gpu);
    m.def("backward", &backward_gpu);
}
