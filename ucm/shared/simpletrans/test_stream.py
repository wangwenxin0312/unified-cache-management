import torch
import ucm_sm_copy


# -----------------------------
# Helper: 构造 kvcache layout 的 device / host 存储
# -----------------------------
def make_kvcache_blocks(num_blocks: int,
                        block_size: int,
                        num_heads: int,
                        head_dim: int,
                        dtype: torch.dtype,
                        device: torch.device):
    """
    构造真实形状的 k_cache:
      k_cache: [num_blocks, block_size, num_heads, head_dim] on device
      slab_host_k: 同形状 pinned host
      ref_k: k_cache 的 CPU 拷贝，用于对比验证
    """
    # device 上的 KV cache
    k_cache = torch.randn(
        (num_blocks, block_size, num_heads, head_dim),
        device=device,
        dtype=dtype,
    )

    # CPU 上的参考值
    ref_k = k_cache.detach().cpu().clone()

    # pinned host buffer
    slab_host_k = torch.empty(
        (num_blocks, block_size, num_heads, head_dim),
        dtype=dtype,
        device=device,
        # pin_memory=True,
    )

    return k_cache, slab_host_k, ref_k


# -----------------------------
# Helper: 在 device 上构造“指针数组”
# -----------------------------
def make_ptr_array(base_ptr: int,
                   block_bytes: int,
                   block_ids,
                   device: torch.device):
    """
    返回 CUDA tensor [len(block_ids)]，dtype=uint64
    每个元素 = base_ptr + block_id * block_bytes
    """
    block_ids = torch.as_tensor(block_ids, dtype=torch.int64, device=device)
    offsets = block_ids * int(block_bytes)
    ptrs = (offsets + int(base_ptr)).to(torch.uint64)
    return ptrs.contiguous()  # 一定要 contiguous


# -----------------------------
# 单个测试 Case：用自己创建的 stream + events 控制
# -----------------------------
def run_one_case_with_stream_and_events(
    block_ids,
    num_blocks=4,
    block_size=128,
    num_heads=8,
    head_dim=128,
    dtype=torch.float16,
    device_str="cuda",
):
    print(f"\n=== 测试 block_ids = {block_ids} ===")
    assert torch.cuda.is_available(), "CUDA is not available"

    device = torch.device(device_str)

    # -------------------------
    # 0) 清空默认流上的残余任务
    # -------------------------
    torch.cuda.synchronize(device)

    # -------------------------
    # 1) 构造 k_cache / slab_host_k / 参考值
    # -------------------------
    k_cache, slab_host_k, ref_k = make_kvcache_blocks(
        num_blocks,
        block_size,
        num_heads,
        head_dim,
        dtype,
        device,
    )

    # 计算每个 block 的元素个数和字节数
    block_elems = block_size * num_heads * head_dim
    elem_size = torch.empty((), dtype=dtype).element_size()
    block_bytes = block_elems * elem_size

    base_dev = k_cache.data_ptr()
    base_host = slab_host_k.data_ptr()

    # -------------------------
    # 2) 创建独立 stream + 事件
    # -------------------------
    stream = torch.cuda.Stream(device=device)
    backend = ucm_sm_copy.TransBackend(int(stream.cuda_stream))

    d2h_start = torch.cuda.Event(enable_timing=True)
    d2h_end = torch.cuda.Event(enable_timing=True)
    h2d_start = torch.cuda.Event(enable_timing=True)
    h2d_end = torch.cuda.Event(enable_timing=True)

    # -------------------------
    # 3) 在 device 上构造指针数组
    # -------------------------
    dev_ptrs = make_ptr_array(base_dev, block_bytes, block_ids, device)
    host_ptrs = make_ptr_array(base_host, block_bytes, block_ids, device)

    # -------------------------
    # 4) D2H: k_cache -> slab_host_k
    # -------------------------
    print("[D2H] launching on custom stream...")

    with torch.cuda.stream(stream):
        d2h_start.record(stream)

        backend.copy_d2h(
            int(dev_ptrs.data_ptr()),   # src_ptrs_addr: device 上的 device 指针数组地址
            int(host_ptrs.data_ptr()),  # dst_ptrs_addr: device 上的 host 指针数组地址
            int(block_bytes),
            int(len(block_ids)),
        )

        d2h_end.record(stream)

    d2h_end.synchronize()
    d2h_ms = d2h_start.elapsed_time(d2h_end)
    print(f"[D2H] done, elapsed = {d2h_ms:.3f} ms")

    # 校验：slab_host_k 是否正确接收了对应 block
    host_flat = slab_host_k.view(-1)  # 按元素展平（dtype 维度）
    ref_flat = ref_k.view(-1)

    for bid in block_ids:
        # 源 block 在 k_cache/ref_k 中的元素范围
        src_start = bid * block_elems
        src_end = src_start + block_elems

        # host 中该 block 位置也是按 bid 摆放（因为指针用的是 base_host + bid * block_bytes）
        host_start = bid * block_elems
        host_end = host_start + block_elems

        if not torch.allclose(host_flat[host_start:host_end], ref_flat[src_start:src_end]):
            raise RuntimeError(f"[D2H] mismatch on block {bid}")

    print("[D2H] OK")

    # -------------------------
    # 5) H2D: slab_host_k -> k_cache
    # -------------------------
    print("[H2D] launching on custom stream...")

    # 清空 device 侧 k_cache
    k_cache.zero_()
    torch.cuda.synchronize(device)

    with torch.cuda.stream(stream):
        h2d_start.record(stream)

        backend.copy_h2d(
            int(host_ptrs.data_ptr()),  # src_ptrs_addr: device 上 host 指针数组
            int(dev_ptrs.data_ptr()),   # dst_ptrs_addr: device 上 device 指针数组
            int(block_bytes),
            int(len(block_ids)),
        )

        h2d_end.record(stream)

    h2d_end.synchronize()
    h2d_ms = h2d_start.elapsed_time(h2d_end)
    print(f"[H2D] done, elapsed = {h2d_ms:.3f} ms")

    # 校验：k_cache 对应 block 是否恢复成 ref_k
    dev_flat = k_cache.detach().cpu().view(-1)

    for bid in block_ids:
        src_start = bid * block_elems
        src_end = src_start + block_elems

        if not torch.allclose(dev_flat[src_start:src_end], ref_flat[src_start:src_end]):
            raise RuntimeError(f"[H2D] mismatch on block {bid}")

    print("[H2D] OK")


# -----------------------------
# 主测试入口
# -----------------------------
def main():
    # case 1: 顺序 block
    run_one_case_with_stream_and_events(block_ids=[0, 1, 2, 3])

    # case 2: 乱序 block
    run_one_case_with_stream_and_events(block_ids=[3, 1, 0])

    # case 3: 子集 block
    run_one_case_with_stream_and_events(block_ids=[2])

    print("\n*** All KV-cache shaped tests passed (custom stream + events) ***")


if __name__ == "__main__":
    main()
