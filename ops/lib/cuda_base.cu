#include <cmath>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_base.cuh"

#define _OPS_INTERNAL_IMPLEMENTATION
#define CUDA_DEVICE_PREFIX __device__
#include "ops.hpp"

#define FULL_MASK 0xffffffff

#define CUDA_ERROR_CHECK(fun)                                                                        \
    do                                                                                               \
    {                                                                                                \
        cudaError_t err = fun;                                                                       \
        if (err != cudaSuccess)                                                                      \
        {                                                                                            \
            fprintf(stderr, "Cuda error %d %s:: %s\n", __LINE__, __func__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                      \
        }                                                                                            \
    } while (0);

#define CUDA_DEVICE_CHECK(device, pointer)                                                                                                              \
    cudaPointerAttributes attributes;                                                                                                                   \
    CUDA_ERROR_CHECK(cudaPointerGetAttributes(&attributes, pointer));                                                                                   \
    if (attributes.devicePointer == NULL || device != attributes.device)                                                                                \
    {                                                                                                                                                   \
        if (attributes.devicePointer == NULL)                                                                                                           \
        {                                                                                                                                               \
            printf("CUDA error %d %s :: %s\n", __LINE__, __func__, "pointer is not resident on a GPU");                                                 \
        }                                                                                                                                               \
        else if (device != attributes.device)                                                                                                           \
        {                                                                                                                                               \
            printf("CUDA error %d %s :: pointer not resident on correct GPU (this: %d, pointer: %d)\n", __LINE__, __func__, device, attributes.device); \
        }                                                                                                                                               \
        exit(EXIT_FAILURE);                                                                                                                             \
    }

/*
    Computes the index for buffer values which are shared across GRID_DIM_Y
*/
__device__ int get_index(int i) { return i * blockDim.y + threadIdx.y; }

__host__ __device__ int32_t find_integer_divisor(int32_t x, int32_t bdim)
{
    return (x + bdim - 1) / bdim;
}

template <typename scalar_t>
__global__ void forward_kernel(
    const scalar_t* restrict X,
    const int64_t n_x,
    const int64_t l_x,
    const scalar_t* restrict Y,
    const int64_t n_y,
    const int64_t l_y,
    const int64_t* restrict receiver_list,
    const int64_t num_receivers,
    const int64_t* restrict neighbour_indices,
    const int64_t num_indices,
    scalar_t* restrict output,
    const int64_t n_output,
    const int64_t l1_output,
    const int64_t l2_output)
{
    int32_t edge_start = neighbour_indices[blockIdx.x];
    int32_t edge_end = 0;

    int32_t node_index = receiver_list[edge_start]; // get the idnex of the node we need to sum into.

    if (blockIdx.x == num_indices - 1) // nnodes -1
    {
        edge_end = n_y - 1; // nedges -1
    }
    else
    {
        edge_end = neighbour_indices[blockIdx.x + 1];
    }

    // check if this node has neighbours
    if (edge_end - edge_start == 0)
    {
        return;
    }

    int32_t feat_start = blockIdx.y * blockDim.x;

    bool valid = feat_start + threadIdx.x < l_x ; //X.size(1);

    __syncthreads();

    // for (int32_t m = threadIdx.y; m < Y.size(1); m += blockDim.y)
    for (int32_t m = threadIdx.y; m < l_y); m += blockDim.y)
    {
        scalar_t tmp_output = 0.0;

        for (int32_t i = edge_start; i < edge_end; i++)
        {

            scalar_t y = Y[(i-1)*l_y + m]; // Y[i][m];
            scalar_t x = 0.0;

            if (valid)
            {
                // x = X[i][feat_start + threadIdx.x];
                x = X[(i-1)*l_x + feat_start + threadIdx.x];
            }

            tmp_output += x * y;
        }

        if (valid)
        {
            // output[node_index][m][feat_start + threadIdx.x] = tmp_output;
            output[(node_index-1)*(l1_output*l2_output) + (m-1)*(l1_output) + feat_start + threadIdx.x] = tmp_output;
        }
    }
}

#define FEATS_PER_BLOCK_Y 32
#define M_PER_BLOCK_X 4

// nx = 4, ny = 32
template <typename scalar_t>
__global__ void backward_dX_kernel(
    const scalar_t* restrict Y,
    const int64_t n_y,
    const int64_t l_y,
    const scalar_t* restrict grad_in, // [nnodes, m, feat]
    const int64_t n_grad_in,
    const int64_t l1_grad_in,
    const int64_t l2_grad_in,
    const int64_t* restrict receiver_list,
    const int64_t num_receivers,
    const int64_t* restrict neighbour_indices,
    const int64_t num_indices,
    const scalar_t* restrict grad_X)
{

    extern __shared__ char buffer[];
    size_t offset = 0;
    scalar_t *buffer_grad_in = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += FEATS_PER_BLOCK_Y * l_y * sizeof(scalar_t);

    int32_t edge_start = neighbour_indices[blockIdx.x];
    int32_t edge_end = 0;

    int32_t node_index = receiver_list[edge_start];

    if (blockIdx.x == num_indices - 1) // nnodes -1
    {
        edge_end = n_y; // nedges -1
    }
    else
    {
        edge_end = neighbour_indices[blockIdx.x + 1];
    }
    int32_t nedges = edge_end - edge_start;

    // check if this node has neighbours
    if (nedges == 0)
    {
        return;
    }
    int global_tid = threadIdx.y * blockDim.x + threadIdx.x; // [0 - 128], ny = 4, nx = 32

    int32_t feat_start = blockIdx.y * FEATS_PER_BLOCK_Y; // each block handles FEATS_PER_BLOCK_Y features

    {
        int nx = 32;
        int tidx = global_tid % nx;
        int tidy = global_tid / nx;
        int ny = (blockDim.y * blockDim.x) / nx;

        for (int m = tidy; m < l_y; m += ny)
        {
            scalar_t val = 0.0;

            if (feat_start + tidx < l2_grad_in)
            {
                // val = grad_in[node_index][m][feat_start + tidx];
                val = grad_in[node_index*l1_grad_in*l2_grad_in + m*l1_grad_in + feat_start + tidx];
            }

            buffer_grad_in[m * FEATS_PER_BLOCK_Y + tidx] = val;
        }
    }

    __syncthreads();
    int niter_m = find_integer_divisor(Y.size(1), blockDim.x);             // 16 / 4
    int niter_gradx = find_integer_divisor(FEATS_PER_BLOCK_Y, blockDim.y); // 32 / 32

    for (int32_t i = edge_start; i < edge_end; i++)
    {
        for (int x_idx = 0; x_idx < niter_gradx; x_idx++)
        {
            int feat = feat_start + x_idx * blockDim.y + threadIdx.y;

            scalar_t tmp_output = 0.0;

            // need to reduce along the m dimension, so we divide threads into groups of 4 or 8 and then do warp reductions across those subgroups.
            for (int y_idx = 0; y_idx < niter_m; y_idx++)
            {
                int m = y_idx * blockDim.x + threadIdx.x;

                scalar_t y = 0.0;
                scalar_t tmp_grad_in = 0.0;

                if (m < l_y && feat < l2_grad_in)
                {
                    tmp_grad_in = buffer_grad_in[m * FEATS_PER_BLOCK_Y + threadIdx.y];
                }

                if (m < l_y)
                {
                    // y = Y[i][m];
                    y = Y[i*n_y + m];
                }

                scalar_t tmp_grad = tmp_grad_in * y;

                for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
                {
                    tmp_grad += __shfl_down_sync(FULL_MASK, tmp_grad, offset);
                }

                tmp_output += tmp_grad;
            }

            if (threadIdx.x == 0 && feat < l2_grad_in)
            {
                // grad_X[i][feat] = tmp_output;
                grad_X[i*n_grad_in + feat] = tmp_output;
            }
        }
    }
}

// ny = 4, nx = 32
template <typename scalar_t>
__global__ void backward_dY_kernel(
    const scalar_t* restrict X,
    const int64_t n_x,
    const int64_t l_x,
    const scalar_t* restrict grad_in, // [nnodes, m, feat]
    const int64_t n_grad_in,
    const int64_t l1_grad_in,
    const int64_t l2_grad_in,
    const int64_t* restrict receiver_list,
    const int64_t num_receivers,
    const int64_t* restrict neighbour_indices,
    const int64_t num_indices,
    const scalar_t* restrict grad_Y)
{

    extern __shared__ char buffer[];
    size_t offset = 0;
    scalar_t *buffer_grad_in = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += X.size(1) * M_PER_BLOCK_X * sizeof(scalar_t); // e.g 128 x 4

    int32_t edge_start = neighbour_indices[blockIdx.x];
    int32_t edge_end = 0;

    int32_t node_index = receiver_list[edge_start];

    if (blockIdx.x == neighbour_indices.size(0) - 1) // nnodes -1
    {
        edge_end = X.size(0); // nedges -1
    }
    else
    {
        edge_end = neighbour_indices[blockIdx.x + 1];
    }
    int32_t nedges = edge_end - edge_start;

    // check if this node has neighbours
    if (nedges == 0)
    {
        return;
    }

    int32_t m_start = blockIdx.y * M_PER_BLOCK_X; // each block handles M_PER_BLOCK_X m indices

    int niter_m = find_integer_divisor(M_PER_BLOCK_X, blockDim.y);
    int niter_x = find_integer_divisor(X.size(1), blockDim.x);

    for (int m_idx = 0; m_idx < niter_m; m_idx++)
    {
        int local_m = m_idx * blockDim.y + threadIdx.y;
        int global_m = m_start + local_m;

        for (int feat = threadIdx.x; feat < X.size(1); feat += blockDim.x)
        {
            scalar_t val = 0.0;

            if (global_m < grad_in.size(1))
            {
                val = grad_in[node_index][global_m][feat];
            }

            buffer_grad_in[local_m * X.size(1) + feat] = val;
        }
    }

    __syncthreads();

    for (int32_t i = edge_start; i < edge_end; i++)
    {

        // need to reduce along the channel dimension

        for (int m_idx = 0; m_idx < niter_m; m_idx++)
        {
            int local_m = m_idx * blockDim.y + threadIdx.y;
            int global_m = m_start + m_idx * blockDim.y + threadIdx.y;

            scalar_t tmp_output = 0.0;

            for (int x_idx = 0; x_idx < niter_x; x_idx++)
            {
                int feat = x_idx * blockDim.x + threadIdx.x;

                scalar_t tmp_grad_in = 0.0;

                if (global_m < grad_in.size(1) && feat < X.size(1))
                {
                    tmp_grad_in = buffer_grad_in[local_m * X.size(1) + feat];
                }

                scalar_t x = 0.0;

                if (feat < X.size(1))
                {
                    x = X[i][feat];
                }

                scalar_t tmp_grad = tmp_grad_in * x;

                for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
                {
                    tmp_grad += __shfl_down_sync(FULL_MASK, tmp_grad, offset);
                }

                tmp_output += tmp_grad;
            }

            if (threadIdx.x == 0 && global_m < grad_in.size(1))
            {
                grad_Y[i][global_m] = tmp_output;
            }
        }
    }
}

#define NEIGHBOUR_NEDGES_PER_BLOCK 512

/*
This function takes a sorted input sender_list, which maps each edge to a node by index, and finds the positions of first occurences

This is required by the CUDA code so we can send all calculations per-node to a single block.

the function loads NEIGHBOUR_NEDGES_PER_BLOCK + 1 elements into shared memory, and then loops through the buffer twice. Once for even boundaries, once for odd boundaries.
*/

__global__ void calculate_neighbours_kernel(
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> sender_list,
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