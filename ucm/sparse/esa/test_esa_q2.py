# test_esa.py
import os
import time
import pathlib
import torch
import pytest

from build_utils import build_shared


def load_module():
    """
    Build & import the extension.
    Use env:
      - BUILD_MODE=debug|release   (default: debug)
      - USE_TORCH_EXTENSION=0|1    (default: 0, we use nvcc build_shared here)
    """
    use_torch = os.environ.get("USE_TORCH_EXTENSION", "0") == "1"
    build_mode = os.environ.get("BUILD_MODE", "debug")

    if use_torch:
        # 如果你要走 torch.utils.cpp_extension.load，请在这里补 debug flags
        from torch.utils.cpp_extension import load as torch_load
        extra_cflags = ["-std=c++17"]
        extra_cuda_cflags = []
        if build_mode != "release":
            extra_cflags += ["-O0", "-g3", "-fno-omit-frame-pointer"]
            extra_cuda_cflags += ["-O0", "-g", "-G", "-lineinfo", "--source-in-ptx", "-DTORCH_USE_CUDA_DSA"]
        else:
            extra_cflags += ["-O3"]
            extra_cuda_cflags += ["-O3", "-lineinfo"]

        return torch_load(
            name="esa_interface",
            sources=["esa_interface.cc", "esa_kernels.cu", "esa_sm_copy.cu"],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            verbose=True,
        )

    # nvcc 手工 build：输出到当前目录 esa_interface.so
    so_path = pathlib.Path(__file__).with_name("esa_interface.so")
    if (not so_path.exists()) or os.environ.get("FORCE_REBUILD", "0") == "1":
        build_shared(["./esa_interface.cc", "./esa_kernels.cu", "./esa_sm_copy.cu"],
                     str(so_path),
                     mode=build_mode)

    import importlib.util
    spec = importlib.util.spec_from_file_location("esa_interface", str(so_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load esa_interface.so")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


esa = load_module()

class style():
    RED = '\033[31m'
    GREEN = '\033[32m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

def print_red(msg):
    print(style.RED + msg + style.RESET)

def print_green(msg):
    print(style.GREEN + msg + style.RESET)

def print_blue(msg):
    print(style.BLUE + msg + style.RESET)

def print_yellow(msg):
    print(style.YELLOW + msg + style.RESET)

def _wait_ready(handle: int, timeout_s: float = 5.0) -> bool:
    t0 = time.time()
    while True:
        st = esa.esa_retrieval_poll(handle)
        if st == 1:
            return True
        if st < 0:
            raise RuntimeError(f"invalid handle: {handle}")
        if time.time() - t0 > timeout_s:
            return False
        time.sleep(0.001)

def test_esa_retrieval_q2(batch_size, num_repre_blocks, num_q_heads):
    """
    目标：从 Python 进入 esa_retrieval_launcher -> retrieval_kernel_bf16 (esa_kernels.cu)
    运行时建议：
      CUDA_LAUNCH_BLOCKING=1 BUILD_MODE=debug cuda-gdb --args python -m pytest -s test_esa.py::test_esa_retrieval_debug
    """
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    dim = 128
    num_k_heads = 8
    total_blocks = num_repre_blocks
    N = total_blocks * 2  # repre_cache rows

    # query = torch.randn(batch_size, num_q_heads, dim, device=device, dtype=dtype)
    # repre_cache = torch.randn(N, num_k_heads, dim, device=device, dtype=dtype)
    
    base_dir="/home/externals/wangwenxin21/unified-cache-management/repre/bs2"
    q_file = "q_req1_bs2.pt"
    repre_file = "req1_bs2.pt"
    query = torch.load(os.path.join(base_dir, q_file))
    repre_cache = torch.load(os.path.join(base_dir, repre_file))
    # indices
    q_index_file="q_index_bs2.pt"
    repre_index_file="repre_index_bs2.pt"
    # repre_index = torch.randint(0, N, (total_blocks,), device=device, dtype=torch.int32)
    # q_index = torch.randint(0, batch_size, (total_blocks,), device=device, dtype=torch.int32)
    repre_index = torch.load(os.path.join(base_dir, repre_index_file))
    q_index = torch.load(os.path.join(base_dir, q_index_file))

    # offsets [B+1]
    bs_offset_file="bs_offset_bs2.pt"
    batch_offset = torch.load(os.path.join(base_dir, bs_offset_file))
    # batch_offset = torch.arange(0, (batch_size + 1) * num_repre_blocks,
    #                             step=num_repre_blocks,
    #                             device=device,
    #                             dtype=torch.int32)

    # outputs
    score = torch.empty(total_blocks, device=device, dtype=dtype)

    # pinned CPU buffers required by C++ checks
    score_cpu = torch.empty(total_blocks, device="cpu", dtype=dtype, pin_memory=True)
    score_sorted_cpu = torch.empty(total_blocks, device="cpu", dtype=dtype, pin_memory=True)
    index_sorted_cpu = torch.empty(total_blocks, device="cpu", dtype=torch.int32, pin_memory=True)

    # pinned CPU tensor for repre_index_cpu (C++ 会用 data_ptr<int32_t>())
    repre_index_cpu = torch.empty(total_blocks, device="cpu", dtype=torch.int32, pin_memory=True)

    # pack structs
    inp = esa.RetrievalInputTensor()
    inp.query = query
    inp.repre_cache = repre_cache
    inp.q_index = q_index
    inp.repre_index = repre_index
    inp.repre_index_cpu = repre_index_cpu
    inp.batch_offset = batch_offset
    inp.batch = batch_size
    inp.s = total_blocks

    out = esa.RetrievalOutputTensor()
    out.score = score
    out.score_cpu = score_cpu
    out.score_sorted_cpu = score_sorted_cpu
    out.index_sorted_cpu = index_sorted_cpu

    esa.esa_retrieval(inp, out)

    start = time.perf_counter_ns()
    esa.esa_retrieval(inp, out)
    torch.cuda.synchronize()
    duration = time.perf_counter_ns() - start
    print_green(f"{' '*4}esa_retrieval host API time: {duration/1e6:.3f} ms")

    def naive_retrieval():
        query_batched = query[q_index].to(torch.float32)
        key = torch.repeat_interleave(repre_cache[repre_index],
                                      num_q_heads//num_k_heads,
                                      dim=1).to(torch.float32)
        score_gt = (query_batched * key).sum(-1).sum(-1).to(dtype)
        index_gt = torch.cat([ repre_index[s:t][score_gt[s:t].argsort(descending=True)] for s,t in zip(batch_offset[:-1], batch_offset[1:]) ])
        return score_gt, index_gt

    start = time.perf_counter_ns()
    score_gt, index_gt = naive_retrieval()
    torch.cuda.synchronize()
    duration = time.perf_counter_ns() - start
    print_red(f"{' '*4}naive_retrieval host API time: {duration/1e6:.3f} ms")

    diff = (score - score_gt[:85]).abs()
    print_blue(f"{' '*4}score diff: {diff.mean():.3f}(mean), {diff.max():.3f}(max)")
    
    print("")
    assert diff.mean() < 1e-3



if __name__ == "__main__":
    # 方便直接 python test_esa.py 跑
    test_esa_retrieval_q2(2, 85, 40)
    # test_esa_repre_debug(16)
