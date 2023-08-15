#include <iostream>
#include <torch/extension.h>
#include <omp.h>


torch::Tensor find_first_occurrences(torch::Tensor scatter_indices, long out_dim) {
    // Finds the positions of the first occurrences within scatter_indices
    
    long scatter_size = scatter_indices.size(0);
    long* scatter_indices_ptr = scatter_indices.data_ptr<long>();
    torch::Tensor first_occurrences = torch::empty({out_dim}, torch::dtype(torch::kLong));
    first_occurrences.fill_(-1);
    long* first_occurrences_ptr = first_occurrences.data_ptr<long>();
    first_occurrences_ptr[scatter_indices_ptr[0]] = 0;

    #pragma omp parallel for
    for (long i = 0; i < scatter_size-1; i++) {
        if (scatter_indices_ptr[i] < scatter_indices_ptr[i+1]) first_occurrences_ptr[scatter_indices_ptr[i+1]] = i+1;
    }

    return first_occurrences;
}


template<typename scalar_t>
torch::Tensor forward_t(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor scatter_indices,
    torch::Tensor first_occurrences,
    long out_dim
) {

    long size_scatter = scatter_indices.size(0);
    long size_a = tensor_a.size(1);
    long size_b = tensor_b.size(1);
    torch::Tensor result = torch::zeros(
        {out_dim, size_a, size_b},
        torch::TensorOptions().device(tensor_a.device()).dtype(tensor_a.dtype())
    );

    scalar_t* result_ptr = result.data_ptr<scalar_t>();
    scalar_t* tensor_a_ptr = tensor_a.data_ptr<scalar_t>();
    scalar_t* tensor_b_ptr = tensor_b.data_ptr<scalar_t>();
    long* scatter_indices_ptr = scatter_indices.data_ptr<long>();
    long* first_occurrences_ptr = first_occurrences.data_ptr<long>();

    #pragma omp parallel for
    for (long idx_out = 0; idx_out < out_dim; idx_out++) {
        long idx_in = first_occurrences_ptr[idx_out];
        if (idx_in < 0) continue;
        while (scatter_indices_ptr[idx_in] == idx_out) {
            for (long idx_a = 0; idx_a < size_a; idx_a++) {
                for (long idx_b = 0; idx_b < size_b; idx_b++) {
                    result_ptr[size_a*size_b*idx_out+size_b*idx_a+idx_b] += tensor_a_ptr[size_a*idx_in+idx_a] * tensor_b_ptr[size_b*idx_in+idx_b];
                }
            }
            idx_in++;
            if (idx_in == size_scatter) break;
        }
    }

    return result;
}


template<typename scalar_t>
std::vector<torch::Tensor> backward_t(
    torch::Tensor grad_output,
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor scatter_indices,
    torch::Tensor first_occurrences,
    long out_dim
) {

    long size_a = tensor_a.size(1);
    long size_b = tensor_b.size(1);
    long size_scatter = scatter_indices.size(0);
    torch::Tensor grad_a = torch::Tensor();
    torch::Tensor grad_b = torch::Tensor();

    scalar_t* grad_output_ptr = grad_output.data_ptr<scalar_t>();
    scalar_t* tensor_a_ptr = tensor_a.data_ptr<scalar_t>();
    scalar_t* tensor_b_ptr = tensor_b.data_ptr<scalar_t>();
    long* scatter_indices_ptr = scatter_indices.data_ptr<long>();
    long* first_occurrences_ptr = first_occurrences.data_ptr<long>();

    
    if (tensor_a.requires_grad()) {
        grad_a = torch::zeros_like(tensor_a);
        scalar_t* grad_a_ptr = grad_a.data_ptr<scalar_t>();

        #pragma omp parallel for
        for (long idx_out = 0; idx_out < out_dim; idx_out++) {
            long idx_in = first_occurrences_ptr[idx_out];
            if (idx_in < 0) continue;
            while (scatter_indices_ptr[idx_in] == idx_out) {
                for (long idx_a = 0; idx_a < size_a; idx_a++) {
                    for (long idx_b = 0; idx_b < size_b; idx_b++) {
                        grad_a_ptr[size_a*idx_in+idx_a] += grad_output_ptr[size_a*size_b*idx_out+size_b*idx_a+idx_b] * tensor_b_ptr[size_b*idx_in+idx_b];
                    }
                }
                idx_in++;
                if (idx_in == size_scatter) break;
            }
        }
    }

    if (tensor_b.requires_grad()) {
        grad_b = torch::zeros_like(tensor_b);
        scalar_t* grad_b_ptr = grad_b.data_ptr<scalar_t>();

        #pragma omp parallel for
        for (long idx_out = 0; idx_out < out_dim; idx_out++) {
            long idx_in = first_occurrences_ptr[idx_out];
            if (idx_in < 0) continue;
            while (scatter_indices_ptr[idx_in] == idx_out) {
                for (long idx_a = 0; idx_a < size_a; idx_a++) {
                    for (long idx_b = 0; idx_b < size_b; idx_b++) {
                        grad_b_ptr[size_b*idx_in+idx_b] += grad_output_ptr[size_a*size_b*idx_out+size_b*idx_a+idx_b] * tensor_a_ptr[size_a*idx_in+idx_a];
                    }
                }
                idx_in++;
                if (idx_in == size_scatter) break;
            }
        }
    }

    return {grad_a, grad_b};
}


torch::Tensor forward(
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor scatter_indices,
    torch::Tensor first_occurrences,
    long out_dim
) {
    // Dispatch type by hand
    if (tensor_a.dtype() == c10::kDouble) {
        return forward_t<double>(tensor_a, tensor_b, scatter_indices, first_occurrences, out_dim);
    } else if (tensor_a.dtype() == c10::kFloat) {
        return forward_t<float>(tensor_a, tensor_b, scatter_indices, first_occurrences, out_dim);
    } else {
        throw std::runtime_error("Unsupported dtype");
    }
}


std::vector<torch::Tensor> backward(
    torch::Tensor grad_output,
    torch::Tensor tensor_a,
    torch::Tensor tensor_b,
    torch::Tensor scatter_indices,
    torch::Tensor first_occurrences,
    long out_dim
) {
    // Dispatch type by hand
    if (tensor_a.dtype() == c10::kDouble) {
        return backward_t<double>(grad_output, tensor_a, tensor_b, scatter_indices, first_occurrences, out_dim);
    } else if (tensor_a.dtype() == c10::kFloat) {
        return backward_t<float>(grad_output, tensor_a, tensor_b, scatter_indices, first_occurrences, out_dim);
    } else {
        throw std::runtime_error("Unsupported dtype");
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ops forward cpu");
    m.def("backward", &backward, "ops backward cpu");
    m.def("find_first_occurrences", &find_first_occurrences, "find first occurrences");
}
