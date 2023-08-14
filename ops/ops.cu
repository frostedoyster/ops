#include <iostream>
#include <torch/extension.h>
#include <torch/script.h>
#include <cuda.h>

using namespace std;
using namespace torch;

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
    extern __shared__ char buffer[];
    size_t offset = 0;
    scalar_t *smem = reinterpret_cast<scalar_t *>(buffer + offset);

    int32_t edge_start = indices[blockIdx.x];
    int32_t edge_end = 0;

    if (blockIdx.x == indices.size(0) - 1)
    {
        edge_end = Y.size(0) - 1;
    }
    else
    {
        edge_end = indices[blockIdx.x + 1];
    }

    // clear out shared memory storage...
    for (int32_t m = threadIdx.y; m < Y.size(1); m += blockDim.y)
    {
        for (int32_t x_id = threadIdx.x; x_id < X.size(1); x_id += blockDim.x)
        {
            smem[m * X.size(1) + x_id] = 0.0;
        }
    }

    __syncthreads();

    for (int32_t i = edge_start; i < edge_end; i++)
    {
        for (int32_t m = threadIdx.y; m < Y.size(1); m += blockDim.y)
        {
            scalar_t y = Y[i][m];

            for (int32_t x_id = threadIdx.x; x_id < X.size(1); x_id += blockDim.x)
            {
                scalar_t x = X[i][x_id];

                smem[m * X.size(1) + x_id] += x * y;
            }
        }
    }

    __syncthreads();

    for (int32_t m = threadIdx.y; m < Y.size(1); m += blockDim.y)
    {
        for (int32_t x_id = threadIdx.x; x_id < X.size(1); x_id += blockDim.x)
        {
            output[blockIdx.x][m][x_id] = smem[m * X.size(1) + x_id];
        }
    }
}

torch::Tensor forward_gpu(torch::Tensor X,
                          torch::Tensor Y,
                          torch::Tensor neighbour_indices,
                          int32_t natoms,
                          int32_t nthreadx,
                          int32_t nthready,
                          int32_t nthreadz)
{

    torch::Tensor output = torch::empty({natoms, Y.size(1), X.size(1)},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    dim3 block_dim(natoms);

    dim3 grid_dim(nthreadx, nthready, 1);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "forward_gpu", ([&]
                                  {
                    size_t total_buff_size = 0;

                    total_buff_size += X.size(1) * Y.size(1) * sizeof(scalar_t);

                    forward_kernel<scalar_t><<<block_dim, grid_dim, total_buff_size>>>(
                        X.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                        Y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                        neighbour_indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                        output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()); }));

    cudaDeviceSynchronize();

    return output;
}

template <typename scalar_t>
__global__ void forward2_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> Y,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> indices,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output)
{
    extern __shared__ char buffer[];
    size_t offset = 0;
    scalar_t *smem = reinterpret_cast<scalar_t *>(buffer + offset);

    int32_t edge_start = indices[blockIdx.x];
    int32_t edge_end = 0;

    if (blockIdx.x == indices.size(0) - 1)
    {
        edge_end = Y.size(0) - 1;
    }
    else
    {
        edge_end = indices[blockIdx.x + 1];
    }

    int32_t feat_start = blockIdx.y * blockDim.x;

    // clear out shared memory storage...
    for (int32_t m = threadIdx.y; m < Y.size(1); m += blockDim.y)
    {
        smem[m * blockDim.x + threadIdx.x] = 0.0;
    }

    __syncthreads();

    for (int32_t i = edge_start; i < edge_end; i++)
    {
        for (int32_t m = threadIdx.y; m < Y.size(1); m += blockDim.y)
        {
            scalar_t y = Y[i][m];

            scalar_t x = 0.0;

            if (feat_start + threadIdx.x < X.size(1))
            {
                x = X[i][feat_start + threadIdx.x];
            }

            smem[m * blockDim.x + threadIdx.x] += x * y;
        }
    }

    __syncthreads();

    for (int32_t m = threadIdx.y; m < Y.size(1); m += blockDim.y)
    {
        if (feat_start + threadIdx.x < X.size(1))
        {
            output[blockIdx.x][m][feat_start + threadIdx.x] = smem[m * blockDim.x + threadIdx.x];
        }
    }
}

torch::Tensor forward2_gpu(torch::Tensor X,
                           torch::Tensor Y,
                           torch::Tensor neighbour_indices,
                           int32_t natoms,
                           int32_t nthreadx,
                           int32_t nthready,
                           int32_t nthreadz)
{

    torch::Tensor output = torch::empty({natoms, Y.size(1), X.size(1)},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    int32_t nby = find_integer_divisor(X.size(1), nthreadx);

    dim3 block_dim(natoms, nby);

    dim3 grid_dim(nthreadx, nthready, 1);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "forward2_gpu", ([&]
                                   {
                    size_t total_buff_size = 0;

                    total_buff_size += nthreadx * Y.size(1) * sizeof(scalar_t);

                    forward2_kernel<scalar_t><<<block_dim, grid_dim, total_buff_size>>>(
                        X.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                        Y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                        neighbour_indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                        output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()); }));

    cudaDeviceSynchronize();

    return output;
}

template <typename scalar_t>
__global__ void forward3_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> X,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> Y,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> indices,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> output)
{

    int32_t edge_start = indices[blockIdx.x];
    int32_t edge_end = 0;

    if (blockIdx.x == indices.size(0) - 1)
    {
        edge_end = Y.size(0) - 1;
    }
    else
    {
        edge_end = indices[blockIdx.x + 1];
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

torch::Tensor forward3_gpu(torch::Tensor X,
                           torch::Tensor Y,
                           torch::Tensor neighbour_indices,
                           int32_t natoms,
                           int32_t nthreadx,
                           int32_t nthready,
                           int32_t nthreadz)
{

    torch::Tensor output = torch::empty({natoms, Y.size(1), X.size(1)},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    int32_t nby = find_integer_divisor(X.size(1), nthreadx);

    dim3 block_dim(natoms, nby);

    dim3 grid_dim(nthreadx, nthready, 1);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "forward3_gpu", ([&]
                                   {
                    size_t total_buff_size = 0;

                    //total_buff_size += nthreadx * Y.size(1) * sizeof(scalar_t);

                    forward3_kernel<scalar_t><<<block_dim, grid_dim, total_buff_size>>>(
                        X.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                        Y.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                        neighbour_indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
                        output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()); }));

    cudaDeviceSynchronize();

    return output;
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

torch::Tensor calculate_neighbours_gpu(torch::Tensor sender_list, int32_t natoms, int32_t nthreadx)
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("calculate_neighbours", &calculate_neighbours_gpu, "computes neighbourlist starts from sender list.");
    m.def("forward", &forward_gpu, "ops forward GPU.");
    m.def("forward2", &forward2_gpu, "ops forward 2 GPU.");
    m.def("forward3", &forward3_gpu, "ops forward 2 GPU.");
}