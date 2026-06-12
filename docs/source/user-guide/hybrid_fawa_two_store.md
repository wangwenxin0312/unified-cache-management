# Qwen Hybrid FAWA 双 Store 卸载实现

本文档说明 `ucm/integration/vllm/hybrid_connector.py` 中
`UCMHybridFAWAConnector` 如何为 Qwen3.6-27B 这类 full-attention +
linear-attention/Mamba-align hybrid model 做 KV/state 卸载。

和 `UCMHybridLinearAttentionConnector` 的单 store whole-page 方案不同，
`UCMHybridFAWAConnector` 把同一个 hybrid raw page 拆成两类语义数据：

- FA store：保存 full-attention KV，即 `[k_or_ssm_state, v_or_padding]` 中
  attention 层真正使用的 K/V 部分。
- WA store：保存 linear-attention state，即 `[conv_state, k_or_ssm_state]`。

这样 attention prefix 和 state boundary 可以分别 lookup、分别 load/dump。

## 适用条件

`UCMHybridFAWAConnector.can_handle_kv_cache_config()` 先复用
`use_hybrid_linear_attention_layout()` 判断是否存在 full-attention 和
Mamba-align 共用 raw tensor 的 hybrid 布局，然后要求所有 KV cache group 的
token block size 相同，并且能被默认 hash block size `384` 整除。

默认配置为：

```text
hash_block_size = 384
token_block_size % 384 == 0
```

以常见 1536-token vLLM aligned block 为例：

```text
token_block_size = 1536
hash_block_size  = 384
一个 vLLM physical block = 4 个 canonical hash block
```

调度和 store key 都按 384 token 边界对齐；物理地址再映射到 1536-token
vLLM block 内的 offset。

## 物理 Layout

`HybridFAWAGroupLayout` 为每个 group 生成 store 可见的 pointer row。
Qwen hybrid raw page 的逻辑布局是：

```text
[conv_state, k_or_ssm_state, v_or_padding]
```

当同一个 raw tensor 同时服务 attention 和 Mamba state 时：

```text
conv_size   = Mamba component 0
ssm_size    = Mamba component 1
k_size      = attention K
v_size      = attention V
middle_size = max(k_size, ssm_size)
tail_size   = raw_page_size - conv_size - middle_size
```

FA store 选择 component `[1, 2]`：

```text
FA payload = [middle_size, tail_size]
           = [K 或 SSM 所在中段, V 或 padding 所在尾段]
```

WA store 选择 component `[0, 1]`：

```text
WA payload = [conv_size, middle_size]
           = [conv state, SSM state]
```

代码优先使用 vLLM 暴露的 direct tensor view。若没有 direct view，则 fallback 到
raw tensor component 计算。raw fallback 中 block stride 是完整 raw page size，
因为 Qwen hybrid page 在 vllm-ascend 中按每个 vLLM block 连续组织：

```text
block N page: [conv, middle, tail]
block N+1 page: [conv, middle, tail]
```

## 384 对齐与地址映射

hash key 每 384 token 一个，但 physical block 可能覆盖 1536 token。因此 FA 和
WA 的地址映射不同。

### FA store

FA KV 是 token-contiguous 的 attention prefix。`_extract_fa_ptr()` 对每个
canonical 384 block 计算它在 1536-token physical block 内的 offset：

```text
token_start  = hash_index * 384
token_offset = token_start % token_block_size
```

如果 `hash_block_size == token_block_size`，直接使用 physical block base。
否则调用 `layout.extract_addrs_with_offsets()`，把 384-token offset 转换成每个
tensor component 的 byte offset。

例子：

```text
token_block_size = 1536
hash boundaries  = 384, 768, 1152, 1536

canonical block 0 -> physical block 0, offset 0
canonical block 1 -> physical block 0, offset 384
canonical block 2 -> physical block 0, offset 768
canonical block 3 -> physical block 0, offset 1152
```

所以 FA store 可以用 4 个 384-key 分段保存同一个 1536 physical block 里的
attention KV。

### WA store

Mamba-align state 不是 token-contiguous KV。它表示某个边界的完整 running state，
不能只取 1536 block 内的 384-token offset。因此 hybrid connector 覆盖了
`_extract_wa_ptr()`：

```text
384-token boundary key -> 完整 [conv_state, ssm_state] state page
pointer 始终指向 state page base
```

代码注释也明确说明：即使 aligned vLLM block 跨 1536 token，384 boundary key
保存的仍是完整 state page，而不是 page 内的一个 token slice。

## 两个 Store 的布局

### FA store

```text
FA store
└── key: canonical 384-token prefix hash
    └── shard 0
        └── attention KV slices for this 384-token segment
```

FA store 需要 prefix-contiguous，因此 lookup 使用 `lookup_on_prefix()`。

### WA store：默认 row-sharded

`hybrid_fawa_wa_state_row_sharding` 默认开启。WA state 会按 layer/state row 拆成多
个独立 key：

```text
WA store
├── key: hash("UCM_HYBRID_FAWA_WA_STATE_ROW", row_id=0, canonical_key)
│   └── shard 0: row 0 的 [conv, ssm]
├── key: hash("UCM_HYBRID_FAWA_WA_STATE_ROW", row_id=1, canonical_key)
│   └── shard 0: row 1 的 [conv, ssm]
└── ...
```

注意 row id 被编码进 key，实际提交给 store 的 `shard_index` 是 0。这和
layerwise 单 store 使用 `shard_index=row_id` 的方式不同。

如果 row-sharded 条件不满足，会 fallback 到 base FAWA 的 wide WA store：

```text
WA store
└── key: canonical 384-token prefix hash
    └── shard 0
        └── all state rows
```

## Hash 设计

请求 token 用 `generate_hash(hash_block_size=384, token_ids, seed)` 生成 canonical
hash chain：

```text
H0 = hash(seed, tokens[0:384])
H1 = hash(H0,   tokens[384:768])
H2 = hash(H1,   tokens[768:1152])
...
```

这些 canonical hashes 同时作为 FA boundary key 和 WA state boundary key 的基础。

TP rank 处理上，hybrid FAWA 不做 base FAWA 的 TP block 分片保存；因为 Qwen
hybrid KV/state tensor 是 TP-sharded 的，每个 rank 都需要保存自己的 shard。
非 0 rank 会把 key 再包装一次：

```text
rank_key = hash("UCM_HYBRID_FAWA_TP_RANK", tp_rank, canonical_key)
```

WA row-sharded key 会先由 row id 和 canonical key 生成 row key，提交 load/dump
时再进入 rank scope。

## Lookup 流程

调度侧 `get_num_new_matched_tokens()` 要求 `num_computed_tokens` 按 384 对齐：

```text
hbm_hit_block_num = num_computed_tokens / 384
external_keys = canonical_hashes[hbm_hit_block_num:]
```

然后 `_lookup_external_hit_blocks()` 执行两阶段 lookup：

1. FA store 先做 `lookup_on_prefix(external_keys)`，得到 attention KV 连续命中的
   最大 prefix。
2. 在 FA 命中范围内，从最长命中边界往前找 WA state。
   - row-sharded WA：对某个 boundary key，必须所有 row key 都存在。
   - wide WA：只检查该 boundary key 是否存在。

返回值是同时满足 FA prefix 和 WA boundary state 的最大 384-block 数。

这里的关键是：FA 必须连续，WA 不要求每个 boundary 连续存在，只要最终 resume
边界的 state 完整存在即可。因为恢复 linear-attention 时只需要命中前缀末端的
state，而不需要中间每个 state。

## 50% 命中时 state block 如何工作

假设请求长度 3072 token，hash block 为 384，vLLM aligned block 为 1536，
并且外部 cache 命中前 50%：

```text
request tokens      = 3072
target hit tokens   = 1536
canonical hit keys  = K0, K1, K2, K3
physical block span = one 1536-token block
```

### Lookup

FA store 需要 `K0..K3` 连续存在。若 `lookup_on_prefix()` 返回 4 个 384 blocks，
说明 attention KV 可恢复到 1536 token。

WA store 只需要检查最后命中边界 `K3` 的完整 state：

```text
row_key_0 = hash(row_id=0, K3)
row_key_1 = hash(row_id=1, K3)
...
```

所有 row key 都存在时，external hit blocks = 4，external hit tokens = 1536。

如果 FA 命中 4 个 blocks，但 WA 的 `K3` state 缺失，lookup 会向前尝试 `K2`、
`K1`、`K0`。找到哪个完整 state，就把外部命中降级到哪个 384 边界；都找不到则
外部命中为 0。

### Load

恢复时 load metadata 包含：

```text
load_keys = [K0, K1, K2, K3]
```

FA load 使用所有 `load_keys`，按 384 offset 把 attention KV 写回 vLLM KV cache：

```text
K0 -> physical block 0 offset 0
K1 -> physical block 0 offset 384
K2 -> physical block 0 offset 768
K3 -> physical block 0 offset 1152
```

WA load 只取最后一个 key：

```text
window_keys = [K3]
```

对于 Mamba-align group，`_slice_group_block_ids_for_reason(..., reason="load")`
会从该 group 的 vLLM block table 末尾找最后一个非 0 block id。原因是恢复请求
时，vLLM 会保留 prefix state 的逻辑位置，同时在尾部给当前 running state 分配
真实物理页；UCM 要把 `K3` 对应的 state 加载到这个当前 running state block。

row-sharded WA load 会把每个 row key 映射到该 row 的 state pointer：

```text
store key = hash(row_id, K3)
ptr       = current running state block 的 row pointer
```

这样前 1536 token 的 linear state 被恢复为一个完整 boundary state，后续 decode
或 prefill continuation 可以从 50% 命中位置继续计算。

### Dump

dump 时 FA 对每个新完成的 384 block 都保存 attention KV。WA row-sharded 路径会
对每个 dump key 保存 state rows：

```text
dump_keys = 新完成的 canonical 384 keys
WA row key = hash(row_id, dump_key)
```

对于 Mamba-align dump，state block id 由 boundary token 所在的 aligned vLLM
block 计算：

```text
boundary_block_idx = boundary_token_idx // token_block_size
```

在 1536-token block 内的 4 个 384 boundary 会映射到同一个 aligned state block。
实际可复用 lookup 会优先选择最长 FA prefix 对应的 WA state；50% 命中恢复时
使用的就是 1536 边界 `K3` 对应的完整 state。

## 与单 Store 方案的差异

- 单 store whole-page 保存完整 hybrid physical page；FAWA 双 store 按语义拆成
  attention KV 和 state。
- FA store 支持 384-token 粒度的 prefix KV 复用。
- WA store 只要求最终 resume boundary 的 state 完整存在；这正是 50% 命中时
  state block 能工作的关键。
- row-sharded WA 把每层 state row 拆成独立 key，避免一个超宽 state row 造成大
  payload。

