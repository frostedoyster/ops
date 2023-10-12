#ifndef OPS_CUDA_BASE_CUH
#define OPS_CUDA_BASE_CUH

#include <vector>
#include <cmath>

__host__ __device__
int32_t find_integer_divisor(int32_t x, int32_t bdim);

template <typename scalar_t>
__global__
void forward_kernel(
    const scalar_t* X,
    const int64_t n_x,
    const int64_t l_x,
    const scalar_t* Y,
    const int64_t n_y,
    const int64_t l_y,
    const int64_t* receiver_list,
    const int64_t num_receivers,
    const int64_t* neighbour_indices,
    const int64_t num_indices,
    scalar_t* output,
    const int64_t n_output,
    const int64_t l1_output,
    const int64_t l2_output);

template <typename scalar_t>
__global__
void backward_dX_kernel(
    const scalar_t* Y,
    const int64_t n_y,
    const int64_t l_y,
    const scalar_t* grad_in, // [nnodes, m, feat]
    const int64_t n_grad_in,
    const int64_t l1_grad_in,
    const int64_t l2_grad_in,
    const int64_t* receiver_list,
    const int64_t num_receivers,
    const int64_t* neighbour_indices,
    const int64_t num_indices,
    const scalar_t* grad_X);

template <typename scalar_t>
__global__
void backward_dY_kernel(
    const scalar_t* X,
    const int64_t n_x,
    const int64_t l_x,
    const scalar_t* grad_in, // [nnodes, m, feat]
    const int64_t n_grad_in,
    const int64_t l1_grad_in,
    const int64_t l2_grad_in,
    const int64_t* receiver_list,
    const int64_t num_receivers,
    const int64_t* neighbour_indices,
    const int64_t num_indices,
    const scalar_t* grad_Y);

__global__
void calculate_neighbours_kernel(
    const int64_t* sender_list,
    const int64_t num_senders,
    int64_t* edge_indices);

#endif //OPS_CUDA_BASE_CUH
