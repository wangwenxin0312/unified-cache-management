# Qwen3.6 混合注意力模型的单 Store 实现说明

本文档总结 `ucm/integration/vllm/ucm_connector.py` 中
`UCMHybridLinearAttentionConnector` 和
`UCMHybridLinearAttentionLayerWiseConnector` 对 Qwen3.6-27B 这类
full-attention + linear-attention/Mamba-align 混合 KV cache 的处理方式。

## 背景

Qwen3.6-27B 这类模型的 vLLM KV cache 不是传统的“每层一个 K/V cache”
形态。full-attention 层和 linear-attention/Mamba-align 层可能共享同一个底层
raw tensor。相同物理页在不同层会被解释成不同语义：

- full-attention 层把页解释成 attention K/V。
- linear-attention/Mamba-align 层把页解释成 conv state、ssm state 和 padding。

UCM 这里没有为 attention KV 和 linear state 分两个 store，而是用一个 UCM
store 保存同一批物理页。核心思想是：调度层按 hybrid KV cache group 分别生成
hash key 和 vLLM block id；worker 侧用 `HybridLinearAttentionLayout` 把这些
逻辑块映射到共享 raw tensor 的真实内存切片；最终所有 group 的 load/dump block
列表被拍平成 `(ucm_block_id, vllm_block_id)`，提交给同一个 store。

## 选择条件

`use_hybrid_linear_attention_layout()` 会扫描 `kv_cache_config.kv_cache_tensors`。
只要某个 shared raw tensor 同时被 `FullAttentionSpec` 和
`MambaSpec(mamba_cache_mode="align")` 使用，就认为需要 hybrid linear attention
layout。外层 `UCMConnector` 的选择顺序中：

- 若 `UCMHybridFAWAConnector` 能处理，则优先走 FA/WA 双 store 版本。
- 否则当 `use_layerwise=True` 且
  `hybrid_linear_attention_layerwise` 默认开启时，选择
  `UCMHybridLinearAttentionLayerWiseConnector`。
- 否则当 `use_hybrid_linear_attention=True` 时，选择
  `UCMHybridLinearAttentionConnector` 作为普通 whole-block 单 store 路径。

## 调度侧：按 group hash，按 LCM 对齐

`UCMHybridLinearAttentionConnector` 继承 `UCMHMAConnector`，所以调度逻辑来自
HMA 基类。

`KVCacheGroupManager` 按 `kv_cache_config.kv_cache_groups` 构造 group 信息：

- 每个 group 有自己的 `group_id`、`block_size`、`sliding_window`、layer 列表和
  hash seed。
- full-attention group 的 `sliding_window=None`。
- linear-attention/Mamba-align group 被当作 sliding-window 类 group，其中
  `MambaSpec(mamba_cache_mode="align")` 有特殊 state hash。
- 所有 group 的 resume 边界按 `lcm_block_size = lcm(all group block_size)` 对齐。

每个 request 会生成 `HMARequestMeta`：

- `group_ucm_block_ids[gid]`：该 group 按自身 block size 生成的 block hash。
- `group_vllm_block_ids[gid]`：该 group 的 vLLM 物理 block id 表。
- 继承字段 `ucm_block_ids` 仍镜像第一个 full-attention group，用于兼容旧路径。
- `hbm_hit_block_num` 和 `total_hit_block_num` 使用 LCM block 计数。

外部命中分两阶段判断：

1. 对每个 full-attention group 调 `store.lookup_on_prefix()`，取所有 full group
   命中 token 数的最小值，再向下对齐到 `lcm_block_size`。
2. 对每个 sliding-window/Mamba-align group 检查 resume 边界处的尾部 state 是否
   存在。普通 sliding-window 用 `store.lookup(tail_block_ids)`；Mamba-align 用
   基于 primary full-attention prefix hash 派生出的 state hash。

因此，一个请求只有在 full-attention prefix 和 linear state 都存在时才算外部
cache 命中。

## 物理布局：HybridLinearAttentionLayout

`HybridLinearAttentionLayout` 是单 store 的关键。它把 vLLM shared raw tensor
拆成 UCM store 可理解的 tensor slice 列表。

### 逻辑 row

每个 `kv_cache_config.kv_cache_tensors` 中的 shared raw tensor 会形成一个
`row_id`：

- `layer_name_to_row[layer_name] = row_id` 记录某个 vLLM layer 属于哪个 shared row。
- `row_tensor_size_lists[row_id]` 记录该 row 内每个物理 slice 的大小。
- `row_shard_sizes[row_id] = sum(row_tensor_size_lists[row_id])`。
- `row_slices[row_id]` 指向 flatten 后的 `base_ptrs/tensor_size_lists` 范围。

这里的 row 不是严格等同于 transformer layer，而是等同于一个 shared cache
tensor。多个 layer 可能映射到同一个 row。

### Ascend/NPU 布局

Ascend 上 shared page 是 component-major 组织：

```text
raw_tensor:
  [所有 block 的 conv_or_padding]
  [所有 block 的 k_or_ssm]
  [所有 block 的 v_or_padding]
```

布局构造时：

- `conv_size` 来自 Mamba spec 的第 1 个 component。
- `ssm_size` 来自 Mamba spec 的第 2 个 component。
- `k_size/v_size` 来自 full-attention spec。
- 中间段大小取 `middle_size = max(k_size, ssm_size)`，保证同一段可容纳 attention K
  或 Mamba SSM。
- 尾段大小为 `tail_size = page_size - conv_size - middle_size`，且必须能容纳
  attention V。

最终一个 row 暴露为 3 个 UCM tensor slice：

```text
row tensor_size_list = [conv_size, middle_size, tail_size]

slice 0: base + 0
slice 1: base + conv_size * num_blocks
slice 2: base + (conv_size + middle_size) * num_blocks

block_stride_list = [conv_size, middle_size, tail_size]
```

所以同一个 `vllm_block_id` 对应的内存地址为：

```text
ptr[slice] = base_ptr[slice] + vllm_block_id * block_stride[slice]
```

### CUDA 布局

CUDA 上一个 physical block 是连续页。此时不拆 component，只暴露成一个 slice：

```text
row tensor_size_list = [page_size]
base_ptr = min(shared_ptrs)
block_stride = page_size
```

同一页由 attention 或 Mamba 代码按不同 view 解释。

## Whole-block connector：UCMHybridLinearAttentionConnector

`UCMHybridLinearAttentionConnector` 本身只覆盖
`_create_kv_cache_layout()`，返回 `HybridLinearAttentionLayout`。其读写行为沿用
`UCMHMAConnector` 与 `UCMDirectConnector`：

- 注册 KV cache 时用 layout 创建一个 store。
- `block_data_size = kv_cache_layout.block_size`。
- load 时把所有要恢复的 `(ucm_block_id, vllm_block_id)` 拍平，调用
  `kv_cache_layout.extract_block_addrs(vllm_block_ids)` 得到每个 block 的所有 slice
  指针，再用 `shard_index=0` 调 `store.load_data()`。
- save 时同样聚合所有 dump block，用 `shard_index=0` 调 `store.dump_data()`。

store 视角下，一个 key 对应一个完整 hybrid physical page：

```text
store key = group-specific UCM block hash
shard_index = 0
payload = flatten(row0 slices, row1 slices, ..., rowN slices)
```

对于 Ascend，payload 是所有 row 的三段 component-major slice 拼接；对于 CUDA，
payload 是所有 row 的连续 page 拼接。由于 key 已经包含 group seed，同一 token
prefix 在不同 group 下不会撞 key。

## Layerwise connector：UCMHybridLinearAttentionLayerWiseConnector

Layerwise 版本仍然只有一个 UCM store，但把 store shard 用作 row 维度：

```text
store key = group-specific UCM block hash
shard_index = row_id
payload = row_id 对应 shared raw tensor 的一页数据
```

### Store 创建

`register_kv_caches()` 中会：

1. 构造 `HybridLinearAttentionLayout`。
2. 收集 `layer_name_to_row` 和所有 `row_ids`。
3. 检查所有 row 的 `row_tensor_size_list` 必须相同。因为只创建一个 row-sharded
   store，store 的 `tensor_size_list` 只能是一种 schema。
4. 用第一个 row 的 `row_tensor_size_list` 创建 store：

```text
tensor_size_list = row_tensor_size_list
shard_size       = sum(row_tensor_size_list)
block_size       = shard_size * (max(row_ids) + 1)
```

这里 `block_size_override` 不是单个 row 的大小，而是让 store 知道同一个 key
下面最多有 `max(row_ids)+1` 个 row shard。真正读写某一行时通过
`shard_index=row_id` 区分。

### Load 流程

`start_load_kv()` 不立即加载全部 row，而是：

- 从 metadata 中收集每个 request 的 load block。
- 过滤 vLLM 的 null block id `0`。
- 对非 TP0 且非 MLA 的 rank，对 key 再做一次 rank scope hash。
- 保存到 `request_data`。
- 预提交前 `hybrid_layerwise_prefetch_rows` 个 row 的 load task，默认 2。

模型执行到某层前，vLLM 调 `wait_for_layer_load(layer_name)`：

1. 通过 `layer_name_to_row` 找到 `row_id`。
2. 若该 row 还未提交，则提交 `store.load_data()`。
3. 等待该 row 的 task 完成。
4. 继续向前预取一个未来 row。

每次 row load 的指针只包含该 row：

```text
ptrs = kv_cache_layout.extract_block_addrs_for_row(vllm_block_ids, row_id)
shard_indexs = [row_id] * len(ucm_block_ids)
store.load_data(ucm_block_ids, shard_indexs, ptrs)
```

这样避免一次性把所有 shared rows 读回 HBM，降低峰值 I/O 和等待时间。

### Save 流程

`save_kv_layer()` 在每层执行后触发，但同一个 row 可能对应多个 layer。为了避免
重复 dump：

- `row_save_layer[row_id]` 取该 row 下 layer id 最大的 layer。
- 只有执行到该 row 的最后一个 layer 时才真正 dump。

dump 数据在第一次保存时由 `_build_dump_transfer_data()` 聚合一次，后续每个 row
复用同一批 `(ucm_block_id, vllm_block_id)`。实际提交时仍按 row 写：

```text
ptrs = kv_cache_layout.extract_block_addrs_for_row(total_vllm_block_ids, row_id)
shard_indexs = [row_id] * len(total_ucm_block_ids)
store.dump_data(total_ucm_block_ids, shard_indexs, ptrs, event_handle)
```

`wait_for_save()` 会等待所有 row 的 dump task，并在 event sync 开启时销毁 event
handle。

## 单 Store 布局总结

### Whole-block 版本

```text
UCM store
└── key: group hash
    └── shard 0
        ├── row 0 slice(s)
        ├── row 1 slice(s)
        └── ...
```

适合实现简单的整块读写。缺点是一个 key 的 payload 是所有 row 的拼接，hybrid
模型上 block 可能很大。

### Layerwise 版本

```text
UCM store
└── key: group hash
    ├── shard row_id=0: row 0 slice(s)
    ├── shard row_id=1: row 1 slice(s)
    └── ...
```

它仍然是一个 store，但借助 `shard_index=row_id` 把不同 shared raw tensor 的页拆
成多个 shard。load/save 都按 row 做，和模型前向的 layer 顺序配合。

## 关键限制

- Layerwise 单 store 要求所有 row 的 `row_tensor_size_list` 完全一致，否则无法用
  一个 row-sharded store 描述所有 row。
- HMA resume 边界必须对齐到所有 group block size 的 LCM。
- sliding-window 的 tail 必须可在 LCM boundary 上无重叠保存；代码要求
  `sliding_window <= lcm_block_size`，且当 `block_size < sliding_window` 时
  `sliding_window` 必须是 `block_size` 的整数倍。
- Mamba-align group 的普通逻辑 block hash 会被空 hash 占位，真正存取的是根据
  full-attention prefix hash 派生的 state hash。
- vLLM block id `0` 是 null block，占位元数据不会被 load/dump。
