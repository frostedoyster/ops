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

#endif //OPS_CUDA_BASE_CUH
