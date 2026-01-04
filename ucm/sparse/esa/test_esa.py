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


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_repre_blocks", [32])
@pytest.mark.parametrize("num_q_heads", [16])
def test_esa_retrieval_debug(batch_size, num_repre_blocks, num_q_heads):
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
    total_blocks = batch_size * num_repre_blocks
    N = total_blocks * 2  # repre_cache rows

    query = torch.randn(batch_size, num_q_heads, dim, device=device, dtype=dtype)
    repre_cache = torch.randn(N, num_k_heads, dim, device=device, dtype=dtype)

    # indices
    repre_index = torch.randint(0, N, (total_blocks,), device=device, dtype=torch.int32)
    q_index = torch.randint(0, batch_size, (total_blocks,), device=device, dtype=torch.int32)

    # offsets [B+1]
    batch_offset = torch.arange(0, (batch_size + 1) * num_repre_blocks,
                                step=num_repre_blocks,
                                device=device,
                                dtype=torch.int32)

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

    handle = esa.esa_retrieval(inp, out)

    # 让 GPU work 都跑完（方便 cuda-gdb 断点稳定）
    torch.cuda.synchronize()

    # 等 CPU worker 完成（否则 score_sorted_cpu / index_sorted_cpu 还没写完）
    ok = _wait_ready(handle, timeout_s=5.0)
    assert ok, "retrieval worker timeout"

    # 清理 ctx（避免 map 越积越多）
    esa.esa_retrieval_cleanup(handle)

    # 最小 sanity：score_cpu 已经有数据（非全 0 / 非 nan）
    sc = out.score_cpu
    assert torch.isfinite(sc).all()
    # 不做数值严格对齐（你调试用），这里只保证链路通了
    print("score_cpu[0:4] =", sc[:4].tolist())
    print("index_sorted_cpu[0:4] =", out.index_sorted_cpu[:4].tolist())


@pytest.mark.parametrize("num_blocks", [16])
def test_esa_repre_debug(num_blocks):
    """
    目标：从 Python 进入 extract_repre_bf16 (esa_kernels.cu)
    """
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    block_size = 128
    dim = 128
    key_rows = num_blocks * 2
    repre_rows = num_blocks * 2

    # key_cache expected by esa_repre: [N, block_size, dim]
    key_cache = torch.randn(key_rows, block_size, dim, device=device, dtype=dtype)
    repre_cache = torch.empty(repre_rows, dim, device=device, dtype=dtype)

    block_table = torch.randint(0, key_rows, (num_blocks,), device=device, dtype=torch.int32)
    repre_index = torch.randint(0, repre_rows, (num_blocks,), device=device, dtype=torch.int32)

    esa.esa_repre(key_cache, repre_cache, block_table, repre_index)
    torch.cuda.synchronize()

    # minimal sanity
    assert torch.isfinite(repre_cache).all()
    print("repre_cache[repre_index[0]] =", repre_cache[repre_index[0]].float()[:4].tolist())


if __name__ == "__main__":
    # 方便直接 python test_esa.py 跑
    test_esa_retrieval_debug(2, 32, 16)
    test_esa_repre_debug(16)
