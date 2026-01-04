#include <torch/extension.h>
#include <thrust/device_ptr.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

// Comparator for ZipIterators: Compares only the first element (the value)
struct ValueComparator {
    __host__ __device__
    bool operator()(const thrust::tuple<int, int>& a, const thrust::tuple<int, int>& b) const {
        return thrust::get<0>(a) < thrust::get<0>(b);
    }
};

std::pair<torch::Tensor, torch::Tensor> filter_unkept_elements(
    torch::Tensor keys, 
    torch::Tensor old_values, 
    torch::Tensor new_values) 
{
    // Ensure all tensors are on the same device and are 1D
    TORCH_CHECK(keys.is_cuda() && old_values.is_cuda() && new_values.is_cuda(), "Tensors must be on CUDA");
    
    int n_new = new_values.size(0);
    int n_old = old_values.size(0);

    // Get raw data pointers and wrap with thrust::device_ptr
    thrust::device_ptr<int> d_keys(keys.data_ptr<int>());
    thrust::device_ptr<int> d_new(new_values.data_ptr<int>());
    thrust::device_ptr<int> d_old(old_values.data_ptr<int>());

    // 1. Sort inputs (Required for set_difference)
    thrust::sort_by_key(d_new, d_new + n_new, d_keys);
    thrust::sort(d_old, d_old + n_old);

    // 2. Prepare output tensors
    auto out_keys = torch::empty_like(new_values);
    auto out_vals = torch::empty_like(new_values);
    thrust::device_ptr<int> d_out_keys(out_keys.data_ptr<int>());
    thrust::device_ptr<int> d_out_vals(out_vals.data_ptr<int>());

    // 3. Zip iterators to keep key and value together
    auto new_zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_new, d_keys));
    auto new_zip_end   = thrust::make_zip_iterator(thrust::make_tuple(d_new + n_new, d_keys + n_new));

    // Dummy sequence for "old" zip to match tuple types
    auto dummy = torch::zeros_like(old_values);
    thrust::device_ptr<int> d_dummy(dummy.data_ptr<int>());
    auto old_zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_old, d_dummy));
    auto old_zip_end   = thrust::make_zip_iterator(thrust::make_tuple(d_old + n_old, d_dummy + n_old));

    auto out_zip_begin = thrust::make_zip_iterator(thrust::make_tuple(d_out_vals, d_out_keys));

    // 4. Perform set difference
    auto result_end = thrust::set_difference(
        new_zip_begin, new_zip_end,
        old_zip_begin, old_zip_end,
        out_zip_begin,
        ValueComparator()
    );

    // 5. Slice outputs to actual size
    int num_unkept = result_end - out_zip_begin;
    return {out_keys.slice(0, 0, num_unkept), out_vals.slice(0, 0, num_unkept)};
}

// Bind to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("filter_unkept", &filter_unkept_elements, "Filter new values not in old values");
}

