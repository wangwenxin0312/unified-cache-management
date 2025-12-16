// cuda_sm_pybind.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "cuda_sm_kernel.h"  // 里边声明了 UC::Trans::CudaSMCopyAsync

namespace py = pybind11;
using Ptr = uintptr_t;

class TransBackend {
public:
    // 构造函数：内部新建一条 CUDA stream（非阻塞）
    TransBackend() : stream_(nullptr), own_stream_(true) {
        auto err = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaStreamCreateWithFlags failed: ") +
                cudaGetErrorString(err));
        }
    }

    // 外部传入已有 stream 指针（uint64）
    TransBackend(uint64_t stream_addr) : stream_(nullptr), own_stream_(false) {
        stream_ = reinterpret_cast<cudaStream_t>(stream_addr);
        if (!stream_) {
            throw std::runtime_error("TransBackend: null stream pointer");
        }
    }

    ~TransBackend() {
        if (own_stream_ && stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    // ============================
    //  Src -> dst
    //  src_ptrs_addr: device 上保存「device 指针数组」的那块内存地址
    //  dst_ptrs_addr: device 上保存「host pinned 指针数组」的那块内存地址
    //  size_bytes:    每个 block 拷贝的字节数（block_bytes）
    //  number:        block 个数（num_blocks）
    // ============================
    void copy_trans(py::object src_ptrs_addr,
                  py::object dst_ptrs_addr,
                  size_t size_bytes,
                  size_t number) {
        // src/dst 是“指向 device 上指针数组的 device 地址” dev_ptrs_k.data_ptr() / dst_ptrs_k.data_ptr()
        Ptr src_val = src_ptrs_addr.cast<Ptr>();
        Ptr dst_val = dst_ptrs_addr.cast<Ptr>();

        auto* src = static_cast<void**>(reinterpret_cast<void*>(src_val));
        auto* dst = static_cast<void**>(reinterpret_cast<void*>(dst_val));

        // 3. 直接调用底层的 CudaSMCopyAsync
        auto err = UC::Trans::CudaSMCopyAsync(
            src,         // void* src[]
            dst,           // void* dst[]
            size_bytes,
            number,
            stream_);

        TORCH_CHECK(err == cudaSuccess,
                    "CudaSMCopyAsync (D2H) failed: ", cudaGetErrorString(err));
    }

    // 同步内部 stream
    void synchronize() {
        auto err = cudaStreamSynchronize(stream_);
        TORCH_CHECK(err == cudaSuccess,
                    "cudaStreamSynchronize failed: ", cudaGetErrorString(err));
    }

    // 导出内部 stream 的地址
    uint64_t get_stream_ptr() const {
        return reinterpret_cast<uint64_t>(stream_);
    }

private:
    cudaStream_t stream_;
    bool own_stream_;  // 是否自己创建并负责销毁 stream_
};

PYBIND11_MODULE(ucm_sm_copy, m) {
    py::class_<TransBackend>(m, "TransBackend")
        .def(py::init<>())
        .def(py::init<uint64_t>(), py::arg("stream_addr"))

        .def("copy_trans",
             &TransBackend::copy_trans,
             py::arg("src_ptrs_addr"),
             py::arg("dst_ptrs_addr"),
             py::arg("size_bytes"),
             py::arg("number"))

        .def("synchronize", &TransBackend::synchronize)
        .def("get_stream_ptr", &TransBackend::get_stream_ptr);
}
