#ifndef OPS_CUDA_BASE_CUH
#define OPS_CUDA_BASE_CUH

#include <vector>
#include <cmath>

template <typename scalar_t>
void forward_kernel(
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
    const int64_t l2_output);

template <typename scalar_t>
void backward_dX_kernel(
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
    const scalar_t* restrict grad_X);

template <typename scalar_t>
void backward_dY_kernel(
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
    const scalar_t* restrict grad_Y);

void calculate_neighbours_kernel(
    const int64_t* restrict sender_list,
    const int64_t num_senders,
    int64_t* restrict edge_indices);

#endif //OPS_CUDA_BASE_CUH
