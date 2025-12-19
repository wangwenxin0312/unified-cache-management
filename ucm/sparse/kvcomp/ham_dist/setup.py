from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_cuda_flags = [
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]

setup(
    name="hamming",
    # packages=find_packages(),
    # include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "hamming",
            ["hamming.cpp", "paged_ham_dist_mla.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": (
                    ["-O3", "--use_fast_math"]
                    + extra_cuda_flags
                ),
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)