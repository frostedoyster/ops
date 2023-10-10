#include <cmath>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_base.hpp"

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
    const restrict scalar_t* X,
    const int64_t n_x;
    const int64_t l_x;
    const restrict scalar_t* Y,
    const int64_t n_y;
    const int64_t l_y;
    const restrict int64_t* receiver_list,
    const restrict int64_t* neighbour_indices,
    restrict scalar_t* output,
    const int64_t n_output,
    const int64_t l1_output,
    const int64_t l2_output)
{
    int32_t edge_start = neighbour_indices[blockIdx.x];
    int32_t edge_end = 0;

    int32_t node_index = receiver_list[edge_start]; // get the idnex of the node we need to sum into.

    if (blockIdx.x == neighbour_indices.size(0) - 1) // nnodes -1
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
            output[(node_index-1)*(l1_output*l2_output) + (m-1)(l1_output) + feat_start + threadIdx.x] = tmp_output;
        }
    }
}
