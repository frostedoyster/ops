#include <iostream>
#include <torch/extension.h>
#include <torch/script.h>
#include <cuda.h>

using namespace std;
using namespace torch::autograd;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

#define FULL_MASK 0xffffffff

__host__ __device__ int32_t find_integer_divisor(int32_t x, int32_t bdim)
{
    return (x + bdim - 1) / bdim;
}

template <typename scalar_t>
__global__ void forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> Y,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> indices,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output)
{

    int32_t edge_start = indices[blockIdx.x];
    int32_t edge_end = 0;

    if (blockIdx.x == indices.size(0) - 1) // nnodes -1
    {
        edge_end = Y.size(0) - 1; // nedges -1
    }
    else
    {
        edge_end = indices[blockIdx.x + 1];
    }

    // check if this node has neighbours
    if (edge_end - edge_start == 0)
    {
        return;
    }

    int32_t feat_start = blockIdx.y * blockDim.x;

    bool valid = feat_start + threadIdx.x < X.size(1);

    __syncthreads();

    for (int32_t m = threadIdx.y; m < Y.size(1); m += blockDim.y)
    {
        scalar_t tmp_output = 0.0;

        for (int32_t i = edge_start; i < edge_end; i++)
        {

            scalar_t y = Y[i][m];
            scalar_t x = 0.0;

            if (valid)
            {
                x = X[i][feat_start + threadIdx.x];
            }

            tmp_output += x * y;
        }

        if (valid)
            output[blockIdx.x][m][feat_start + threadIdx.x] = tmp_output;
    }
}

torch::Tensor forward_gpu(torch::Tensor X,
                          torch::Tensor Y,
                          torch::Tensor neighbour_indices,
                          int64_t natoms,
                          int64_t nthreadx,
                          int64_t nthready,
                          int64_t nthreadz)
{

    CHECK_INPUT(X);
    CHECK_INPUT(Y);
    CHECK_INPUT(neighbour_indices);

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

                    forward_kernel<scalar_t><<<block_dim, grid_dim, total_buff_size>>>(
                        X.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                        Y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                        neighbour_indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                        output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()); }));

    cudaDeviceSynchronize();

    return output;
}

template <typename scalar_t>
__global__ void backward1_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> Y,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_in, // [nnodes, m, feat]
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> indices,
    const bool requires_grad_X,
    const bool requires_grad_Y,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_X,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_Y)
{

    extern __shared__ char buffer[];
    size_t offset = 0;
    scalar_t *buffer_dX = reinterpret_cast<scalar_t *>(buffer + offset);

    int32_t edge_start = indices[blockIdx.x];
    int32_t edge_end = 0;

    if (blockIdx.x == indices.size(0) - 1) // nnodes -1
    {
        edge_end = Y.size(0) - 1; // nedges -1
    }
    else
    {
        edge_end = indices[blockIdx.x + 1];
    }
    int32_t nedges = edge_end - edge_start;

    // check if this node has neighbours
    if (nedges == 0)
    {
        return;
    }

    int32_t feat_start = blockIdx.y * blockDim.x;

    bool valid = feat_start + threadIdx.x < X.size(1);

    __syncthreads();

    for (int32_t m = threadIdx.y; m < Y.size(1); m += blockDim.y)
    {
        scalar_t tmp_grad_in = grad_in[blockIdx.x][m][feat_start];

        for (int32_t i = edge_start; i < edge_end; i++)
        {

            if (requires_grad_X)
            {
                // need to sum over m dimension

                buffer_dX[threadIdx.x] = 0.0;

                scalar_t y = Y[i][m];

                scalar_t tmp_grad_x = tmp_grad_in * y;

                atomicAdd(&buffer_dX[threadIdx.x], tmp_grad_x);

                __syncwarp();

                if (threadIdx.y == 0)
                {
                    grad_X[i][feat_start + threadIdx.x] = buffer_dX[threadIdx.x];
                }
            }

            if (requires_grad_Y)
            {
                scalar_t x = 0.0;

                if (valid)
                {
                    x = X[i][feat_start + threadIdx.x];
                }

                scalar_t tmp_grad_y = tmp_grad_in * x;

                // need to warp reduce over X dimension
                for (int32_t offset = blockDim.x / 2; offset > 0; offset /= 2)
                {
                    tmp_grad_y += __shfl_down_sync(FULL_MASK, tmp_grad_y, offset);
                }

                if (threadIdx.x == 0)
                {
                    grad_Y[i][m] = tmp_grad_y;
                }
            }
        }
    }
}

std::vector<torch::Tensor> backward1_gpu(torch::Tensor X,
                                         torch::Tensor Y,
                                         torch::Tensor grad_in,
                                         torch::Tensor neighbour_indices,
                                         int64_t natoms,
                                         int64_t nthreadx,
                                         int64_t nthready,
                                         int64_t nthreadz)
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

    int32_t nby = find_integer_divisor(X.size(1), nthreadx);

    dim3 block_dim(natoms, nby);

    dim3 grid_dim(nthreadx, nthready, 1);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "backward1_gpu", ([&]
                                    {
        size_t total_buff_size = 0;

        total_buff_size += nthreadx * sizeof(scalar_t);

            backward1_kernel<scalar_t><<<block_dim, grid_dim, total_buff_size>>>(
                X.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                Y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_in.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                neighbour_indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                X.requires_grad(),
                Y.requires_grad(),
                gradX.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
                gradY.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>()); }));

    cudaDeviceSynchronize();

    return {gradX, gradY};
}

template <typename scalar_t>
__global__ void backward2_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> Y,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_in, // [nnodes, m, feat]
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> indices,
    const bool requires_grad_X,
    const bool requires_grad_Y,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_X,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_Y)
{

    extern __shared__ char buffer[];
    size_t offset = 0;
    scalar_t *buffer_dX = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += blockDim.x * sizeof(scalar_t);

    scalar_t *buffer_dY = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += 128 * blockDim.y * sizeof(scalar_t);

    int32_t edge_start = indices[blockIdx.x];
    int32_t edge_end = 0;

    if (blockIdx.x == indices.size(0) - 1) // nnodes -1
    {
        edge_end = Y.size(0) - 1; // nedges -1
    }
    else
    {
        edge_end = indices[blockIdx.x + 1];
    }
    int32_t nedges = edge_end - edge_start;

    // check if this node has neighbours
    if (nedges == 0)
    {
        return;
    }

    int32_t feat = blockIdx.y * blockDim.x + threadIdx.x;

    bool valid = feat < X.size(1);

    __syncthreads();

    for (int32_t m = threadIdx.y; m < Y.size(1); m += blockDim.y)
    {
        scalar_t tmp_grad_in = 0.0;

        if (valid)
            tmp_grad_in = grad_in[blockIdx.x][m][feat];

        for (int32_t i = edge_start; i < edge_end; i++)
        {

            if (requires_grad_X)
            {
                // need to sum over m dimension

                buffer_dX[threadIdx.x] = 0.0;

                scalar_t y = Y[i][m];

                scalar_t tmp_grad_x = tmp_grad_in * y;

                atomicAdd(&buffer_dX[threadIdx.x], tmp_grad_x);

                __syncwarp();

                if (threadIdx.y == 0 && valid)
                {
                    grad_X[i][feat] = buffer_dX[threadIdx.x];
                }
            }

            if (requires_grad_Y)
            {
                scalar_t x = 0.0;

                if (valid)
                {
                    x = X[i][feat];
                }

                scalar_t tmp_grad_y = tmp_grad_in * x;

                // need to warp reduce over X dimension
                for (int32_t offset = blockDim.x / 2; offset > 0; offset /= 2)
                {
                    tmp_grad_y += __shfl_down_sync(FULL_MASK, tmp_grad_y, offset);
                }

                if (threadIdx.x == 0)
                {
                    // buffer_dY[threadIdx.y * nedges + (i - edge_start)] = tmp_grad_y;
                    buffer_dY[(i - edge_start) * blockDim.y + threadIdx.y] = tmp_grad_y;
                }
            }
        }

        __syncwarp();

        for (int i = threadIdx.y; i < nedges; i += blockDim.x)
        {
            // grad_Y[edge_start + i][m] = buffer_dY[threadIdx.y * nedges + i];
            grad_Y[edge_start + i][m] = buffer_dY[i * blockDim.y + threadIdx.y];
        }
    }
}

std::vector<torch::Tensor> backward2_gpu(torch::Tensor X,
                                         torch::Tensor Y,
                                         torch::Tensor grad_in,
                                         torch::Tensor neighbour_indices,
                                         int32_t natoms,
                                         int32_t nthreadx,
                                         int32_t nthready,
                                         int32_t nthreadz)
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

    int32_t nby = find_integer_divisor(X.size(1), nthreadx);

    dim3 block_dim(natoms, nby);

    dim3 grid_dim(nthreadx, nthready, 1);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "backward2_gpu", ([&]
                                    {
        size_t total_buff_size = 0;

        total_buff_size += nthreadx * sizeof(scalar_t);
        total_buff_size += 128 * nthready * sizeof(scalar_t);
            backward2_kernel<scalar_t><<<block_dim, grid_dim, total_buff_size>>>(
                X.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                Y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_in.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                neighbour_indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                X.requires_grad(),
                Y.requires_grad(),
                gradX.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
                gradY.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>()); }));

    cudaDeviceSynchronize();

    return {gradX, gradY};
}

#define NEDGES_PER_BLOCK_BWD 256

template <typename scalar_t>
__global__ void backward3_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> Y,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_in,    // [nnodes, m, feat]
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> sender_list, // [nedges]
    const bool requires_grad_X,
    const bool requires_grad_Y,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_X,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_Y)
{

    extern __shared__ char buffer[];
    size_t offset = 0;
    // scalar_t *buffer_dX = reinterpret_cast<scalar_t *>(buffer + offset);
    // offset += blockDim.x * sizeof(scalar_t);

    // scalar_t *buffer_dY = reinterpret_cast<scalar_t *>(buffer + offset);
    // offset += 128 * blockDim.y * sizeof(scalar_t);

    int edge_start = blockIdx.x * NEDGES_PER_BLOCK_BWD;
    int32_t feat = blockIdx.y * blockDim.x + threadIdx.x;

    bool valid = feat < X.size(1);

    for (int id = threadIdx.z; id < NEDGES_PER_BLOCK_BWD; id += blockDim.z)
    {
        int edge_id = edge_start + id;

        if (edge_id < X.size(0))
        {

            int sender_id = sender_list[edge_id];

            for (int m = threadIdx.y; m < Y.size(1); m += blockDim.y)
            {
                scalar_t tmp_grad = 0.0;

                if (valid)
                    tmp_grad = grad_in[sender_id][m][feat];

                scalar_t x = 0.0;

                if (valid)
                    x = X[edge_id][feat];

                scalar_t tmp_dy = x * tmp_grad; // sum this over x

                for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
                {
                    tmp_dy += __shfl_down_sync(FULL_MASK, tmp_dy, offset);
                }

                if (threadIdx.x == 0)
                {
                    grad_Y[edge_id][m] = tmp_dy;
                }
            }

            int tid = threadIdx.y * blockDim.x + threadIdx.x;

            int ty = tid / 4; // [0-32]
            int tx = tid % 4;

            scalar_t tmp_out = 0.0;

            for (int tx = tid % 4; tx < Y.size(1); tx += 4)
            {
                scalar_t y = Y[edge_id][tx];

                scalar_t tmp_dx = y * grad_in[sender_id][tx][blockIdx.x * 32 + ty];

                for (int offset = 4 / 2; offset > 0; offset /= 2)
                {
                    tmp_dx += __shfl_down_sync(FULL_MASK, tmp_dx, offset);
                }

                tmp_out += tmp_dx;
            }

            if (tx == 0)
            {
                grad_X[edge_id][blockIdx.x * 32 + ty] = tmp_out;
            }
        }
    }
}

std::vector<torch::Tensor> backward3_gpu(torch::Tensor X,
                                         torch::Tensor Y,
                                         torch::Tensor grad_in,
                                         torch::Tensor sender_list,
                                         int32_t natoms,
                                         int32_t nthreadx,
                                         int32_t nthready,
                                         int32_t nthreadz)
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

    int32_t nbx = find_integer_divisor(X.size(0), NEDGES_PER_BLOCK_BWD);
    int32_t nby = find_integer_divisor(X.size(1), nthreadx);

    dim3 block_dim(nbx, nby);

    dim3 grid_dim(nthreadx, nthready, nthreadz);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "backward3_gpu", ([&]
                                    {
        size_t total_buff_size = 0;

        //total_buff_size += nthreadx * sizeof(scalar_t);
        //total_buff_size += 128 * nthready * sizeof(scalar_t);
            backward3_kernel<scalar_t><<<block_dim, grid_dim, total_buff_size>>>(
                X.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                Y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_in.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                sender_list.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                X.requires_grad(),
                Y.requires_grad(),
                gradX.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>(),
                gradY.packed_accessor32<scalar_t,2, torch::RestrictPtrTraits>()); }));

    cudaDeviceSynchronize();

    return {gradX, gradY};
}

#define NEIGHBOUR_NEDGES_PER_BLOCK 512

/*
This function takes a sorted input sender_list, which maps each edge to a node by index, and finds the positions of first occurences

This is required by the CUDA code so we can send all calculations per-node to a single block.

the function loads NEIGHBOUR_NEDGES_PER_BLOCK + 1 elements into shared memory, and then loops through the buffer twice. Once for even boundaries, once for odd boundaries.
*/

__global__ void calculate_neighbours_kernel(const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> sender_list,
                                            torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> edge_indices)
{
    extern __shared__ char buffer[];
    size_t offset = 0;
    int32_t *smem = reinterpret_cast<int32_t *>(buffer + offset);

    int32_t block_start = blockIdx.x * NEIGHBOUR_NEDGES_PER_BLOCK;

    int32_t nedges = sender_list.size(0);

    // load all elements of senderlist needed by block into shared memory
    for (int32_t i = threadIdx.x; i < NEIGHBOUR_NEDGES_PER_BLOCK + 1; i += blockDim.x)
    {
        int32_t idx = block_start + i;

        if (idx < nedges)
        {
            smem[i] = sender_list[idx];
        }
    }

    __syncthreads();

    // deal with even boundaries
    for (int32_t i = 2 * threadIdx.x; i < NEIGHBOUR_NEDGES_PER_BLOCK; i += 2 * blockDim.x)
    {
        int32_t idx = block_start + i;

        if (idx + 1 < nedges)
        {
            int32_t loc1 = smem[i];
            int32_t loc2 = smem[i + 1];

            if (loc1 != loc2)
            {
                edge_indices[loc2] = idx + 1;
            }
        }
    }

    // deal with odd boundaries
    for (int32_t i = 2 * threadIdx.x + 1; i < NEIGHBOUR_NEDGES_PER_BLOCK + 1; i += 2 * blockDim.x)
    {
        int32_t idx = block_start + i;

        if (idx + 1 < nedges)
        {
            int32_t loc1 = smem[i];
            int32_t loc2 = smem[i + 1];

            if (loc1 != loc2)
            {
                edge_indices[loc2] = idx + 1;
            }
        }
    }

    // deal with 0th element specifically, so we dont need to use torch::zeros
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        edge_indices[0] = 0;
    }
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

/*PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("calculate_neighbours", &calculate_neighbours_gpu, "computes neighbourlist starts from sender list.");
    m.def("forward", &forward_gpu, "ops forward GPU.");
    m.def("backward1", &backward1_gpu, "ops backward GPU.");
    m.def("backward2", &backward2_gpu, "ops backward GPU.");
    m.def("backward3", &backward3_gpu, "ops backward GPU.");
}*/

class OuterProductAutograd : public Function<OuterProductAutograd>
{
public:
    static torch::Tensor forward(
        AutogradContext *ctx,
        torch::Tensor X,
        torch::Tensor Y,
        const torch::Tensor sender_list,
        const int64_t natoms,
        const int64_t nthreadx,
        const int64_t nthready,
        const int64_t nthreadz)
    {

        torch::Tensor node_indices = calculate_neighbours_gpu(sender_list, natoms, 64);

        auto result = forward_gpu(X, Y, node_indices, natoms, nthreadx, nthready, nthreadz);

        if (X.requires_grad() || Y.requires_grad())
        {
            ctx->saved_data["nthreadx"] = nthreadx;
            ctx->saved_data["nthready"] = nthready;
            ctx->saved_data["nthreadz"] = nthreadz;
            ctx->saved_data["natoms"] = natoms;

            ctx->save_for_backward({X, Y, sender_list, node_indices});
        }

        return result;
    }

    static variable_list backward(AutogradContext *ctx, variable_list grad_outputs)
    {

        auto saved_variables = ctx->get_saved_variables();

        auto X = saved_variables[0];
        auto Y = saved_variables[1];
        auto sender_list = saved_variables[2];
        auto node_indices = saved_variables[3];

        int nthreadx = ctx->saved_data["nthreadx"].toInt();
        int nthready = ctx->saved_data["nthready"].toInt();
        int nthreadz = ctx->saved_data["nthreadz"].toInt();
        int natoms = ctx->saved_data["natoms"].toInt();

        /*std::vector<torch::Tensor> backward1_gpu(torch::Tensor X,
                                                 torch::Tensor Y,
                                                 torch::Tensor grad_in,
                                                 torch::Tensor neighbour_indices,
                                                 int32_t natoms,
                                                 int32_t nthreadx,
                                                 int32_t nthready,
                                                 int32_t nthreadz) */
        //{
        auto result = backward1_gpu(X, Y, grad_outputs[0], node_indices, natoms, nthreadx, nthready, nthreadz);

        torch::Tensor undef;

        return {result[0], result[1], undef, undef, undef, undef, undef};
    }
};

torch::Tensor outer_product(torch::Tensor X, torch::Tensor Y,
                            torch::Tensor sender_list, int64_t natoms, int64_t nthreadx, int64_t nthready, int64_t nthreadz)
{
    return OuterProductAutograd::apply(X, Y, sender_list, natoms, nthreadx, nthready, nthreadz);
}

TORCH_LIBRARY(unweighted_outer_product, m)
{
    m.def("calculate_neighbours", &calculate_neighbours_gpu);
    m.def("outer_product", &outer_product);
}