import os
import sysconfig
import subprocess
import torch

def build_shared(src_files, target, mode = "release"):
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    cuda_inc = os.path.join(cuda_home, "include")
    cuda_lib = os.path.join(cuda_home, "lib64")
    if not os.path.isdir(cuda_inc) or not os.path.isdir(cuda_lib):
        raise SystemExit(f"CUDA not found. Set CUDA_HOME or install to {cuda_home}")

    py_inc = sysconfig.get_paths()["include"]

    # Torch include/library paths
    t_inc = include_paths()  # e.g., [.../torch/include, .../torch/include/torch/csrc/api/include]
    t_lib = library_paths()  # e.g., [.../torch/lib]

    # ABI flag must match the one PyTorch was built with
    cxx11_abi = getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", 1)
    abi_macro = f"-D_GLIBCXX_USE_CXX11_ABI={int(cxx11_abi)}"
    target_name = os.path.splitext(os.path.basename(target))[0]
    print("target_name: ", target_name)

    print("==== nvcc_compile")
    cmd = [
        "nvcc",
        "-std=c++17",
        "--compiler-options", "-fPIC",   # same as -Xcompiler -fPIC
        "-shared",

        # include
        "-I" + py_inc,
        "-I" + cuda_inc,
        *[f"-I{p}" for p in t_inc],
        # ABI macro
        abi_macro,
        f"-DTORCH_EXTENSION_NAME={target_name}",
        # libs
        "-L" + cuda_lib,
        *[f"-L{p}" for p in t_lib],

        # rpath
        "-Xlinker", "-rpath", "-Xlinker", cuda_lib,
        *[arg for p in t_lib for arg in ("-Xlinker", "-rpath", "-Xlinker", p)],
        # link against torch and CUDA runtime
        "-lc10",
        "-lc10_cuda",
        "-ltorch_cpu",
        "-ltorch_cuda",
        "-ltorch",
        "-ltorch_python",
    "-lcudart",
    ]
    if mode == "release":
        # 可保留 lineinfo（有行号但不能单步到 device 级别）
        print("[compile mode]]", mode)
        cmd += [
            "-O3",
            "-lineinfo",
            "--use_fast_math",
        ]
        # host 优化也会开，问题不大
        cmd += ["-Xcompiler", "-O3"]
    else:
        # 关键：host + device 都要 debug
        cmd += [
            # device debug: 可单步 kernel
            "-G",               # device debug，关闭大部分 device 优化
            "-O0",
            "-lineinfo",
            "--source-in-ptx",  # 便于源码映射/诊断
            "-g",               # 生成 host 侧 debug（nvcc 会传给 host 编译器）
            "-Xcompiler", "-O0",
            "-Xcompiler", "-g3",
            "-Xcompiler", "-fno-omit-frame-pointer",
            "-Xcompiler", "-rdynamic",
            "-UNDEBUG",         # 确保不走 NDEBUG
            "-DTORCH_USE_CUDA_DSA",  # device-side assert（你原来有）
        ]

        # 可选：让 cuda-gdb 更快遇到 launch（调试时常用）
        # cmd += ["-DDEBUG", "-D_GLIBCXX_ASSERTIONS"]

    assert isinstance(src_files, (list, tuple))
    cmd.extend(src_files)
    cmd.extend(["-o", target])
    print("Building so with:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    build_shared(["./esa_interface.cc", "./esa_kernels_simple.cu", "./esa_sm_copy.cu"],"esa_interface.so",mode=os.environ.get("BUILD_MODE", "debug"))
