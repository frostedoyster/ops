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

__host__ __device__ int find_integer_divisor(int x, int bdim)
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
    offset += blockDim.x * sizeof(scalar_t);

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
        for (int32_t channel = threadIdx.x; channel < X.size(1); channel += blockDim.x)
        {
            smem[m * X.size(1) + channel] = 0.0;
        }
    }

    __syncthreads();

    for (int32_t i = edge_start; i < edge_end; i++)
    {
        for (int32_t m = threadIdx.y; m < Y.size(1); m += blockDim.y)
        {
            scalar_t y = Y[i][m];

            for (int32_t channel = threadIdx.x; channel < X.size(1); channel += blockDim.x)
            {
                scalar_t x = X[i][channel];

                smem[m * X.size(1) + channel] += x * y;
            }
        }
    }

    __syncthreads();

    for (int32_t m = threadIdx.y; m < Y.size(1); m += blockDim.y)
    {
        for (int32_t channel = threadIdx.x; channel < X.size(1); channel += blockDim.x)
        {
            output[blockIdx.x][m][channel] = smem[m * X.size(1) + channel];
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

#define NEIGHBOUR_NEDGES_PER_BLOCK 512

/*
This function takes a sorted input sender_list, which maps each edge to a node by index, and outputs the "boundaries" when the index pattern changes

This is required by the CUDA code so we can send all calculations per-node to a single block.

the function loads 1024 + 1 elements into shared memory, and then loops through the buffer twice. Once for even boundaries, once for odd boundaries.
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
    for (int i = threadIdx.x; i < NEIGHBOUR_NEDGES_PER_BLOCK + 1; i += blockDim.x)
    {
        int32_t idx = block_start + i;

        if (idx < nedges)
        {
            smem[i] = sender_list[idx];
        }
    }

    __syncthreads();

    // deal with even boundaries
    for (int i = 2 * threadIdx.x; i < NEIGHBOUR_NEDGES_PER_BLOCK; i += 2 * blockDim.x)
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
    for (int i = 2 * threadIdx.x + 1; i < NEIGHBOUR_NEDGES_PER_BLOCK + 1; i += 2 * blockDim.x)
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

    int nbx = find_integer_divisor(sender_list.size(0), NEIGHBOUR_NEDGES_PER_BLOCK);

    dim3 block_dim(nbx);

    // printf("block dim: %d\n", nbx);

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
}