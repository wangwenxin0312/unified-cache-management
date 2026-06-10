import math
import os
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from ucm.integration.vllm.device import create_device
from ucm.integration.vllm.hma_connector import (
    FAWARequestDispatchMeta,
    FAWARequestMeta,
    FAWADumpTask,
    FAWALoadTask,
    KVCacheGroupMeta,
    UCMFAWAConnector,
    UCMFAWAConnectorMetadata,
)
from ucm.integration.vllm.ucm_connector import (
    _ensure_cache_store_buffer_capacity,
    _attention_component_sizes,
    _mamba_component_sizes,
    layer_name_to_kv_cache_spec,
    use_hybrid_linear_attention_layout,
)
from ucm.logger import init_logger
from ucm.sparse.utils import round_up
from ucm.store.factory_v1 import UcmConnectorFactoryV1

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.request import Request

logger = init_logger(__name__)


class HybridFAWAGroupLayout:
    """Store-visible slices for Qwen hybrid attention/state pages.

    Qwen3.6-style hybrid layouts can back full-attention KV cache and linear
    attention state cache with the same raw page. The raw page is logically:

        [conv_state, k_or_ssm_state, v_or_padding]

    The FA store should persist only the attention KV part
    [k_or_ssm_state, v_or_padding], while the WA store should persist the
    linear-attention state [conv_state, k_or_ssm_state].
    """

    def __init__(
        self,
        kvcaches: dict[str, torch.Tensor],
        vllm_config: "VllmConfig",
        kv_cache_config: "KVCacheConfig",
        group_id: int,
        component_kind: str,
    ) -> None:
        self.kvcaches = kvcaches
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.group_id = group_id
        self.component_kind = component_kind
        self.num_blocks = kv_cache_config.num_blocks
        self.layer_name_to_kv_cache_spec = layer_name_to_kv_cache_spec(
            kv_cache_config
        )
        group_spec = kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
        self.group_token_block_size = _block_size_from_spec(group_spec)
        self._build_layout()

    def _collect_shared_tensor_info(
        self,
        raw_tensor,
    ) -> tuple[list["KVCacheSpec"], list[int]]:
        specs = []
        ptrs = []
        for layer_name in raw_tensor.shared_by:
            specs.extend(self.layer_name_to_kv_cache_spec.get(layer_name, []))
            kv_layer = self.kvcaches.get(layer_name)
            if kv_layer is None:
                continue
            if isinstance(kv_layer, torch.Tensor):
                ptrs.append(kv_layer.data_ptr())
            elif isinstance(kv_layer, (tuple, list)):
                ptrs.extend(
                    tensor.data_ptr()
                    for tensor in kv_layer
                    if isinstance(tensor, torch.Tensor)
                )
            else:
                logger.warning(f"unsupported hybrid kv_layer type: {type(kv_layer)}")
        return specs, ptrs

    def _component_layout(
        self,
        base: int,
        component_sizes: list[int],
    ) -> tuple[list[int], list[int]]:
        """Return per-component base pointers and block strides.

        Qwen hybrid raw pages are laid out as one contiguous page per vLLM
        block:

            [conv_state, k_or_ssm_state, v_or_padding]

        This matches the actual as_strided views built by vllm-ascend for
        mamba state tensors, so the component stride is the full raw page size
        even on NPU.
        """

        page_size = sum(component_sizes)
        offsets = []
        running = 0
        for size in component_sizes:
            offsets.append(running)
            running += size
        return [base + offset for offset in offsets], [page_size] * len(
            component_sizes
        )

    def _select_components(
        self,
        specs: list["KVCacheSpec"],
        raw_page_size: int,
    ) -> Optional[tuple[list[int], list[int]]]:
        attn_specs = [spec for spec in specs if isinstance(spec, FullAttentionSpec)]
        mamba_specs = [spec for spec in specs if isinstance(spec, MambaSpec)]

        if self.component_kind == "attention" and not attn_specs:
            return None
        if self.component_kind == "state" and not mamba_specs:
            return None

        if attn_specs and mamba_specs:
            conv_size, ssm_size = _mamba_component_sizes(mamba_specs[0])[:2]
            k_size, v_size = _attention_component_sizes(attn_specs[0])
            middle_size = max(k_size, ssm_size)
            tail_size = raw_page_size - conv_size - middle_size
            if tail_size <= 0:
                raise ValueError(
                    "Invalid hybrid FAWA page layout: "
                    f"raw_page_size={raw_page_size}, conv_size={conv_size}, "
                    f"middle_size={middle_size}, tail_size={tail_size}"
                )
            if self.component_kind == "attention":
                if tail_size < v_size:
                    raise ValueError(
                        "Hybrid FAWA attention tail cannot hold V cache: "
                        f"tail_size={tail_size}, v_size={v_size}"
                    )
                return [conv_size, middle_size, tail_size], [1, 2]
            return [conv_size, middle_size, tail_size], [0, 1]

        if self.component_kind == "attention" and attn_specs:
            return [raw_page_size], [0]

        if self.component_kind == "state" and mamba_specs:
            mamba_sizes = _mamba_component_sizes(mamba_specs[0])
            if len(mamba_sizes) < 2:
                raise ValueError(
                    "Hybrid FAWA state layout needs at least two Mamba "
                    f"components, got {mamba_sizes}."
                )
            conv_size, ssm_size = mamba_sizes[:2]
            tail_size = raw_page_size - conv_size - ssm_size
            if tail_size < 0:
                raise ValueError(
                    "Invalid hybrid FAWA state page layout: "
                    f"raw_page_size={raw_page_size}, conv_size={conv_size}, "
                    f"ssm_size={ssm_size}"
                )
            if tail_size == 0:
                return [conv_size, ssm_size], [0, 1]
            return [conv_size, ssm_size, tail_size], [0, 1]

        return None

    def _build_layout(self) -> None:
        if self._build_direct_view_layout():
            return
        self._build_raw_tensor_layout()

    def _append_tensor_view(
        self,
        tensor: torch.Tensor,
        base_ptrs: list[int],
        block_strides: list[int],
        tensor_sizes: list[int],
        tensor_token_strides: list[int],
        view_meta: list[dict[str, object]],
        layer_name: str,
        component_idx: int,
    ) -> None:
        element_size = tensor.element_size()
        if tensor.dim() < 1:
            raise ValueError(
                f"Hybrid FAWA tensor view for {layer_name} is scalar-shaped."
            )
        tensor_size = int(math.prod(tensor.shape[1:]) * element_size)
        token_stride = (
            int(tensor.stride(1) * element_size)
            if tensor.dim() > 1
            else tensor_size // self.group_token_block_size
        )
        base_ptrs.append(int(tensor.data_ptr()))
        block_strides.append(int(tensor.stride(0) * element_size))
        tensor_sizes.append(tensor_size)
        tensor_token_strides.append(token_stride)
        view_meta.append(
            {
                "layer_name": layer_name,
                "component_kind": self.component_kind,
                "component_idx": component_idx,
                "shape": tuple(tensor.shape),
                "stride": tuple(tensor.stride()),
                "dtype": str(tensor.dtype),
            }
        )

    def _append_attention_view(
        self,
        tensor: torch.Tensor,
        tensor_size: int,
        base_ptrs: list[int],
        block_strides: list[int],
        tensor_sizes: list[int],
        tensor_token_strides: list[int],
        view_meta: list[dict[str, object]],
        layer_name: str,
        component_idx: int,
    ) -> None:
        element_size = tensor.element_size()
        token_stride = (
            int(tensor.stride(1) * element_size)
            if tensor.dim() > 1
            else tensor_size // self.group_token_block_size
        )
        base_ptrs.append(int(tensor.data_ptr()))
        block_strides.append(int(tensor_size))
        tensor_sizes.append(int(tensor_size))
        tensor_token_strides.append(token_stride)
        view_meta.append(
            {
                "layer_name": layer_name,
                "component_kind": self.component_kind,
                "component_idx": component_idx,
                "shape": tuple(tensor.shape),
                "stride": tuple(tensor.stride()),
                "logical_tensor_size": int(tensor_size),
                "dtype": str(tensor.dtype),
            }
        )

    def _finalize_layout(
        self,
        base_ptrs: list[int],
        block_strides: list[int],
        tensor_sizes: list[int],
        tensor_token_strides: list[int],
        view_meta: list[dict[str, object]],
    ) -> None:
        self.base_ptrs = np.asarray(base_ptrs, dtype=np.uint64)
        self.block_strides = np.asarray(block_strides, dtype=np.uint64)
        self.tensor_sizes = np.asarray(tensor_sizes, dtype=np.uint64)
        self.tensor_block_sizes = np.asarray(
            [self.group_token_block_size] * len(tensor_sizes),
            dtype=np.uint64,
        )
        self.tensor_sizes_per_token = (
            self.tensor_sizes // self.tensor_block_sizes
        ).astype(np.uint64, copy=False)
        self.tensor_token_strides = np.asarray(
            tensor_token_strides, dtype=np.uint64
        )
        self.view_meta = view_meta
        logger.info(
            f"Hybrid FAWA group layout: group_id={self.group_id}, "
            f"kind={self.component_kind}, ptrs={len(base_ptrs)}, "
            f"tensor_sizes={tensor_sizes}"
        )

    def _build_direct_view_layout(self) -> bool:
        base_ptrs: list[int] = []
        block_strides: list[int] = []
        tensor_sizes: list[int] = []
        tensor_token_strides: list[int] = []
        view_meta: list[dict[str, object]] = []

        group = self.kv_cache_config.kv_cache_groups[self.group_id]
        for layer_name in group.layer_names:
            specs = self.layer_name_to_kv_cache_spec.get(layer_name, [])
            kv_layer = self.kvcaches.get(layer_name)
            if kv_layer is None:
                continue

            if self.component_kind == "attention":
                attn_specs = [
                    spec for spec in specs if isinstance(spec, FullAttentionSpec)
                ]
                if not attn_specs:
                    continue
                attn_sizes = _attention_component_sizes(attn_specs[0])
                if isinstance(kv_layer, (tuple, list)) and len(kv_layer) >= 2:
                    for component_idx, tensor in enumerate(kv_layer[:2]):
                        if isinstance(tensor, torch.Tensor):
                            self._append_attention_view(
                                tensor,
                                attn_sizes[component_idx],
                                base_ptrs,
                                block_strides,
                                tensor_sizes,
                                tensor_token_strides,
                                view_meta,
                                layer_name,
                                component_idx,
                            )
                    continue
                if isinstance(kv_layer, torch.Tensor):
                    if kv_layer.dim() == 5 and kv_layer.shape[0] >= 2:
                        views = (kv_layer[0], kv_layer[1])
                    elif kv_layer.dim() == 4 and kv_layer.shape[1] == 2:
                        views = (kv_layer[:, 0], kv_layer[:, 1])
                    else:
                        views = ()
                    for component_idx, tensor in enumerate(views):
                        self._append_attention_view(
                            tensor,
                            attn_sizes[component_idx],
                            base_ptrs,
                            block_strides,
                            tensor_sizes,
                            tensor_token_strides,
                            view_meta,
                            layer_name,
                            component_idx,
                        )

            elif self.component_kind == "state":
                if not any(isinstance(spec, MambaSpec) for spec in specs):
                    continue
                if isinstance(kv_layer, (tuple, list)) and len(kv_layer) >= 2:
                    for component_idx, tensor in enumerate(kv_layer[:2]):
                        if isinstance(tensor, torch.Tensor):
                            self._append_tensor_view(
                                tensor,
                                base_ptrs,
                                block_strides,
                                tensor_sizes,
                                tensor_token_strides,
                                view_meta,
                                layer_name,
                                component_idx,
                            )

        if not base_ptrs:
            return False
        self._finalize_layout(
            base_ptrs,
            block_strides,
            tensor_sizes,
            tensor_token_strides,
            view_meta,
        )
        return True

    def _build_raw_tensor_layout(self) -> None:
        base_ptrs: list[int] = []
        block_strides: list[int] = []
        tensor_sizes: list[int] = []
        tensor_token_strides: list[int] = []
        view_meta: list[dict[str, object]] = []

        for raw_tensor in self.kv_cache_config.kv_cache_tensors:
            if not raw_tensor.shared_by:
                continue
            specs, shared_ptrs = self._collect_shared_tensor_info(raw_tensor)
            if not shared_ptrs:
                continue
            if raw_tensor.size % self.num_blocks != 0:
                raise ValueError(
                    "Invalid hybrid FAWA raw tensor size: "
                    f"raw_size={raw_tensor.size}, num_blocks={self.num_blocks}"
                )
            raw_page_size = raw_tensor.size // self.num_blocks
            selected = self._select_components(specs, raw_page_size)
            if selected is None:
                continue

            component_sizes, component_indices = selected
            component_ptrs, component_strides = self._component_layout(
                min(shared_ptrs),
                component_sizes,
            )
            for component_idx in component_indices:
                size = int(component_sizes[component_idx])
                base_ptrs.append(int(component_ptrs[component_idx]))
                block_strides.append(int(component_strides[component_idx]))
                tensor_sizes.append(size)
                tensor_token_strides.append(size // self.group_token_block_size)
                view_meta.append(
                    {
                        "shared_by": tuple(raw_tensor.shared_by),
                        "component_kind": self.component_kind,
                        "component_idx": component_idx,
                        "component_size": size,
                        "raw_page_size": raw_page_size,
                    }
                )

        if not base_ptrs:
            raise ValueError(
                f"Hybrid FAWA group {self.group_id} has empty "
                f"{self.component_kind} layout."
            )

        self._finalize_layout(
            base_ptrs,
            block_strides,
            tensor_sizes,
            tensor_token_strides,
            view_meta,
        )

    def extract_addrs_with_offsets(
        self,
        block_ids: np.ndarray,
        group_token_block_size: int,
        offsets: np.ndarray,
    ) -> np.ndarray:
        physical_token_offsets = (
            offsets[:, None]
            * self.tensor_block_sizes[None, :]
            // group_token_block_size
        )
        return (
            block_ids[:, None] * self.block_strides[None, :]
            + physical_token_offsets * self.tensor_token_strides[None, :]
            + self.base_ptrs[None, :]
        ).astype(np.uint64, copy=False)

    def extract_addrs(self, block_ids: np.ndarray) -> np.ndarray:
        return (
            block_ids[:, None] * self.block_strides[None, :] + self.base_ptrs[None, :]
        ).astype(np.uint64, copy=False)

    def segment_tensor_size_list(
        self,
        logical_tokens: int,
        group_token_block_size: int,
    ) -> list[int]:
        if self.component_kind == "state":
            return self.tensor_sizes.tolist()

        tensor_tokens = (
            self.tensor_block_sizes * logical_tokens // group_token_block_size
        )
        return (self.tensor_sizes_per_token * tensor_tokens).tolist()

    @property
    def tensor_block_size(self) -> int:
        return int(self.group_token_block_size)


class UCMHybridFAWAConnector(UCMFAWAConnector):
    """FA/WA two-store connector for Qwen hybrid attention + state cache."""

    DEFAULT_HASH_BLOCK_SIZE = 384

    @classmethod
    def can_handle_kv_cache_config(
        cls, kv_cache_config: Optional["KVCacheConfig"]
    ) -> bool:
        if not use_hybrid_linear_attention_layout(kv_cache_config):
            return False
        block_sizes = {
            _block_size_from_spec(group.kv_cache_spec)
            for group in kv_cache_config.kv_cache_groups
        }
        return (
            len(block_sizes) == 1
            and next(iter(block_sizes)) % cls.DEFAULT_HASH_BLOCK_SIZE == 0
        )

    def _init_group_metas(self) -> None:
        groups = self._kv_cache_config.kv_cache_groups
        self.hash_block_size = int(
            self.launch_config.get(
                "hybrid_fawa_hash_block_size", self.DEFAULT_HASH_BLOCK_SIZE
            )
        )
        self.block_size = self.hash_block_size
        self.fa_group_ids, self.window_group_ids = [], []
        self.mamba_align_group_ids: set[int] = set()

        for group_id, group in enumerate(groups):
            spec = _sample_spec(group.kv_cache_spec)
            token_block_size = _block_size_from_spec(group.kv_cache_spec)
            if token_block_size % self.hash_block_size != 0:
                raise ValueError(
                    "Hybrid FAWA hash block size must divide the aligned "
                    f"vLLM block size: group_id={group_id}, "
                    f"token_block_size={token_block_size}, "
                    f"hash_block_size={self.hash_block_size}"
                )
            if isinstance(spec, FullAttentionSpec):
                self.fa_group_ids.append(group_id)
                tail_tokens = self.hash_block_size
            elif isinstance(spec, MambaSpec):
                self.window_group_ids.append(group_id)
                tail_tokens = self.hash_block_size
                if spec.mamba_cache_mode == "align":
                    self.mamba_align_group_ids.add(group_id)
            else:
                logger.warning(
                    f"Skip unsupported hybrid FAWA group {group_id}: "
                    f"{type(spec).__name__}"
                )
                continue

            tail_blocks = max(math.ceil(tail_tokens / token_block_size), 1)
            self.group_metas[group_id] = KVCacheGroupMeta(
                group_id=group_id,
                token_block_size=token_block_size,
                tail_blocks=tail_blocks,
                tail_tokens=tail_tokens,
            )

        if not self.fa_group_ids or not self.window_group_ids:
            raise ValueError(
                "Hybrid FAWA requires at least one full-attention group and "
                "one state group."
            )
        self._wa_state_row_count_hint = sum(
            len(groups[group_id].layer_names) for group_id in self.window_group_ids
        )

    def _base_store_config(self, store_suffix: str):
        name, module_path, config = super()._base_store_config(store_suffix)
        unique_id = f"{self.engine_id}_hybrid_fawa_{store_suffix}"
        config["unique_id"] = unique_id
        backends = config.get("storage_backends")
        if isinstance(backends, list):
            layout_namespace = f"qwen_hybrid_fawa_{self.hash_block_size}_{unique_id}"
            namespaced_backends: list[str] = []
            for backend in backends:
                backend_path = os.path.join(str(backend), layout_namespace)
                os.makedirs(backend_path, exist_ok=True)
                namespaced_backends.append(backend_path)
            config["storage_backends"] = namespaced_backends
        return name, module_path, config

    def _wa_state_row_sharding_enabled(self) -> bool:
        return bool(
            self.launch_config.get("hybrid_fawa_wa_state_row_sharding", True)
        )

    def _build_wa_state_row_specs(
        self,
        group_layouts: dict[int, HybridFAWAGroupLayout],
    ) -> list[tuple[int, slice, list[int]]]:
        specs: list[tuple[int, slice, list[int]]] = []
        for group_id in self.window_group_ids:
            layout = group_layouts.get(group_id)
            if layout is None or getattr(layout, "component_kind", None) != "state":
                continue

            view_meta = getattr(layout, "view_meta", [])
            start = 0
            tensor_count = len(layout.tensor_sizes)
            while start < tensor_count:
                layer_name = (
                    view_meta[start].get("layer_name")
                    if start < len(view_meta) and isinstance(view_meta[start], dict)
                    else None
                )
                end = start + 1
                if layer_name is not None:
                    while (
                        end < tensor_count
                        and end < len(view_meta)
                        and isinstance(view_meta[end], dict)
                        and view_meta[end].get("layer_name") == layer_name
                    ):
                        end += 1
                elif start + 1 < tensor_count:
                    # Raw fallback layouts expose state components in conv/ssm pairs.
                    end = start + 2

                row_slice = slice(start, end)
                row_sizes = [
                    int(size) for size in layout.tensor_sizes[row_slice].tolist()
                ]
                specs.append((group_id, row_slice, row_sizes))
                start = end
        return specs

    def _create_wa_row_sharded_store(
        self,
        tensor_size_list: list[int],
        row_count: int,
        cpu_affinity_cores: Optional[list[int]] = None,
    ):
        name, module_path, config = self._base_store_config("wa")
        if self._role == KVConnectorRole.WORKER:
            row_shard_size = round_up(sum(tensor_size_list), 4096)
            config["device_id"] = self.local_rank
            config["tensor_size_list"] = tensor_size_list
            config["shard_size"] = row_shard_size
            config["block_size"] = row_shard_size
            _ensure_cache_store_buffer_capacity(config, row_shard_size)
            config["local_rank_size"] = self.tp_size if self.is_mla else 1
            if cpu_affinity_cores:
                config["cpu_affinity_cores"] = list(cpu_affinity_cores)
        logger.info(
            "create hybrid FAWA WA row-sharded "
            f"{name} with row_count={row_count}, "
            f"config: {self._summarize_store_config(config)}"
        )
        return UcmConnectorFactoryV1.create_connector(name, config, module_path)

    def _create_wa_store(
        self,
        group_layouts: Optional[dict[int, HybridFAWAGroupLayout]],
        cpu_affinity_cores: Optional[list[int]] = None,
    ):
        self._wa_state_row_sharded = False
        self._wa_state_row_specs: list[tuple[int, slice, list[int]]] = []

        if not self._wa_state_row_sharding_enabled():
            return super()._create_wa_store(group_layouts, cpu_affinity_cores)

        if self._role != KVConnectorRole.WORKER:
            self._wa_state_row_sharded = True
            return self._create_wa_row_sharded_store([], 0, cpu_affinity_cores)

        if group_layouts is None:
            raise RuntimeError("Worker hybrid FAWA WA store needs layouts.")

        row_specs = self._build_wa_state_row_specs(group_layouts)
        if not row_specs:
            logger.warning(
                "Hybrid FAWA WA row-sharded layout is empty; fallback to wide WA store."
            )
            return super()._create_wa_store(group_layouts, cpu_affinity_cores)

        row_tensor_size_list = list(row_specs[0][2])
        for group_id, row_slice, tensor_size_list in row_specs:
            if tensor_size_list != row_tensor_size_list:
                logger.warning(
                    "Hybrid FAWA WA row-sharded rows must share one tensor layout; "
                    "fallback to wide WA store: "
                    f"group_id={group_id}, row_slice={row_slice}, "
                    f"tensor_size_list={tensor_size_list}, "
                    f"expected={row_tensor_size_list}"
                )
                return super()._create_wa_store(group_layouts, cpu_affinity_cores)

        self._wa_state_row_sharded = True
        self._wa_state_row_specs = row_specs
        logger.info(
            "Hybrid FAWA WA row-sharded layout: "
            f"rows={len(row_specs)}, "
            f"row_tensor_size_list={row_tensor_size_list}, "
            f"row_shard_size={sum(row_tensor_size_list)}"
        )
        return self._create_wa_row_sharded_store(
            row_tensor_size_list,
            len(row_specs),
            cpu_affinity_cores,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.kv_caches = kv_caches
        self.device = create_device()

        enable_affinity = os.getenv("VLLM_CPU_AFFINITY") == "1"
        worker_cores, store_cores = (
            self.device.split_cores(self.local_rank)
            if enable_affinity
            else (None, None)
        )

        for group_id in self.fa_group_ids:
            self.group_layouts[group_id] = HybridFAWAGroupLayout(
                kv_caches,
                self._vllm_config,
                self._kv_cache_config,
                group_id,
                "attention",
            )
        for group_id in self.window_group_ids:
            self.group_layouts[group_id] = HybridFAWAGroupLayout(
                kv_caches,
                self._vllm_config,
                self._kv_cache_config,
                group_id,
                "state",
            )

        self.store = self._create_fa_store(self.group_layouts, store_cores)
        self.fa_store = self.store
        self.wa_store = self._create_wa_store(self.group_layouts, store_cores)

        if worker_cores:
            try:
                os.sched_setaffinity(0, worker_cores)
                logger.info(
                    f"[VLLM CPU Affinity] Worker bound to cores {worker_cores}"
                )
            except Exception as e:
                logger.warning(f"Failed to bind worker: {e}")

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if num_computed_tokens % self.hash_block_size != 0:
            raise RuntimeError(
                f"Hybrid FAWA requires aligned computed tokens, got "
                f"{num_computed_tokens} with block size {self.hash_block_size}."
            )
        hbm_hit_block_num = num_computed_tokens // self.hash_block_size
        canonical_hashes = self.generate_hash(
            self.hash_block_size, request.all_token_ids, self._seed
        )

        if self.persist_token_threshold > request.num_tokens:
            return 0, False

        external_keys = canonical_hashes[hbm_hit_block_num:]
        if not external_keys:
            return 0, False

        try:
            external_hit_blocks = self._lookup_external_hit_blocks(external_keys)
        except Exception as e:
            external_hit_blocks = 0
            logger.error(
                f"request {request.request_id} hybrid FAWA lookup error. "
                f"{type(e).__name__}: {e}"
            )

        total_hit_block_num = hbm_hit_block_num + external_hit_blocks
        external_hit_tokens = external_hit_blocks * self.hash_block_size
        num_total_hit_tokens = total_hit_block_num * self.hash_block_size
        if num_total_hit_tokens == request.num_tokens:
            external_hit_tokens -= 1

        self.requests_meta[request.request_id] = FAWARequestMeta(
            ucm_block_ids=canonical_hashes,
            hbm_hit_block_num=hbm_hit_block_num,
            total_hit_block_num=total_hit_block_num,
            num_token_ids=len(request.all_token_ids),
            token_processed=num_total_hit_tokens,
        )
        logger.info(
            f"Hybrid FAWA request_id: {request.request_id}, "
            f"total_blocks_num: {len(canonical_hashes)}, "
            f"hit hbm: {hbm_hit_block_num}, "
            f"hit external: {external_hit_blocks}"
        )
        return external_hit_tokens, False

    def _slice_group_block_ids_for_reason(
        self,
        group_id: int,
        group_block_ids: list[int],
        window_boundary_token_idx: np.ndarray,
        reason: str,
    ) -> list[int]:
        if reason == "dump" and group_id in self.mamba_align_group_ids:
            group_meta = self.group_metas[group_id]
            boundary_block_idx = window_boundary_token_idx // group_meta.token_block_size
            return np.asarray(group_block_ids)[boundary_block_idx].tolist()
        if (
            reason == "load"
            and group_id in self.mamba_align_group_ids
            and group_block_ids
        ):
            for block_id in reversed(group_block_ids):
                if block_id != 0:
                    return [block_id]
            return []
        return self._slice_group_block_ids(
            group_id,
            group_block_ids,
            window_boundary_token_idx,
        )

    def _extract_wa_ptr(self, store_keys, vllm_ids):
        """Build state-store pointer rows.

        Qwen hybrid state pages are not token-contiguous slices. A 384-token
        boundary key stores the complete [conv_state, ssm_state] page, so the
        pointer should stay at the state page base even though the aligned
        vLLM block spans 1536 tokens.
        """

        all_ptrs = []
        for group_id in self.window_group_ids:
            layout = self.group_layouts.get(group_id)
            if layout is None:
                continue
            block_ids = np.asarray(vllm_ids[group_id], dtype=np.uint64)
            if block_ids.size == 0:
                continue
            all_ptrs.append(layout.extract_addrs(block_ids))

        return np.concatenate(all_ptrs, axis=1)

    def _wa_state_row_count(self) -> int:
        row_specs = getattr(self, "_wa_state_row_specs", [])
        if row_specs:
            return len(row_specs)
        return int(getattr(self, "_wa_state_row_count_hint", 0))

    def _wa_state_row_store_key(self, row_id: int, store_key: bytes) -> bytes:
        cache = getattr(self, "_wa_state_row_store_key_cache", None)
        if cache is None:
            cache = {}
            self._wa_state_row_store_key_cache = cache
        cache_key = (row_id, store_key)
        row_key = cache.get(cache_key)
        if row_key is None:
            if len(cache) > 65536:
                cache.clear()
            row_key = self.request_hasher(
                (b"UCM_HYBRID_FAWA_WA_STATE_ROW", row_id, store_key)
            )
            cache[cache_key] = row_key
        return row_key

    def _wa_state_row_store_keys(self, store_key: bytes) -> list[bytes]:
        return [
            self._wa_state_row_store_key(row_id, store_key)
            for row_id in range(self._wa_state_row_count())
        ]

    def _build_wa_state_row_task_data(
        self,
        store_keys: list[bytes],
        vllm_ids,
    ) -> tuple[list[bytes], list[int], np.ndarray]:
        task_keys: list[bytes] = []
        shard_indices: list[int] = []
        ptr_rows: list[np.ndarray] = []
        group_ptr_cache: dict[int, np.ndarray] = {}

        for shard_index, (group_id, row_slice, _) in enumerate(
            self._wa_state_row_specs
        ):
            layout = self.group_layouts.get(group_id)
            if layout is None:
                raise RuntimeError(
                    f"Hybrid FAWA WA row-sharded layout misses group {group_id}."
                )
            block_ids = np.asarray(vllm_ids[group_id], dtype=np.uint64)
            if len(store_keys) == 1 and block_ids.size > 1:
                block_ids = block_ids[-1:]
            if block_ids.size != len(store_keys):
                raise RuntimeError(
                    "Hybrid FAWA WA row-sharded pointer rows do not match keys: "
                    f"group_id={group_id}, keys={len(store_keys)}, "
                    f"block_ids={block_ids.size}"
                )

            if group_id not in group_ptr_cache:
                group_ptr_cache[group_id] = layout.extract_addrs(block_ids)
            group_ptrs = group_ptr_cache[group_id][:, row_slice]
            for key, ptr_row in zip(store_keys, group_ptrs):
                task_keys.append(self._wa_state_row_store_key(shard_index, key))
                shard_indices.append(0)
                ptr_rows.append(ptr_row)

        if not ptr_rows:
            raise RuntimeError("Hybrid FAWA WA row-sharded task is empty.")
        return task_keys, shard_indices, np.asarray(ptr_rows, dtype=np.uint64)

    def _lookup_external_hit_blocks(self, external_keys: list[bytes]) -> int:
        if not getattr(self, "_wa_state_row_sharded", False):
            return super()._lookup_external_hit_blocks(external_keys)
        if self.fa_store is None:
            raise RuntimeError("FA store is not initialized.")
        if self.wa_store is None:
            raise RuntimeError("WA store is not initialized.")

        fa_hit_blocks = self.fa_store.lookup_on_prefix(external_keys) + 1
        if fa_hit_blocks <= 0:
            return 0

        row_count = self._wa_state_row_count()
        for hit_blocks in range(fa_hit_blocks, 0, -1):
            key = external_keys[hit_blocks - 1]
            if row_count <= 0:
                continue

            first_row_key = self._wa_state_row_store_key(0, key)
            if not self.wa_store.lookup([first_row_key])[0]:
                continue

            row_keys = [
                self._wa_state_row_store_key(row_id, key)
                for row_id in range(1, row_count)
            ]
            if not row_keys or all(self.wa_store.lookup(row_keys)):
                return hit_blocks
        return 0

    def _rank_scoped_store_keys(self, keys: list[bytes]) -> list[bytes]:
        if self.tp_rank == 0:
            return keys
        cache = getattr(self, "_rank_scoped_store_key_cache", None)
        if cache is None:
            cache = {}
            self._rank_scoped_store_key_cache = cache

        scoped_keys = []
        for key in keys:
            scoped_key = cache.get(key)
            if scoped_key is None:
                if len(cache) > 65536:
                    cache.clear()
                scoped_key = self.request_hasher(
                    (b"UCM_HYBRID_FAWA_TP_RANK", self.tp_rank, key)
                )
                cache[key] = scoped_key
            scoped_keys.append(scoped_key)
        return scoped_keys

    def _submit_load_task(
        self,
        request_id: str,
        label: str,
        store,
        keys: list[bytes],
        ptrs: np.ndarray,
        anchor_vllm_block_ids: set[int],
    ) -> FAWALoadTask:
        return super()._submit_load_task(
            request_id,
            label,
            store,
            self._rank_scoped_store_keys(keys),
            ptrs,
            anchor_vllm_block_ids,
        )

    def _submit_load_task_with_shards(
        self,
        request_id: str,
        label: str,
        store,
        keys: list[bytes],
        shard_indices: list[int],
        ptrs: np.ndarray,
        anchor_vllm_block_ids: set[int],
    ) -> FAWALoadTask:
        task = store.load_data(
            self._rank_scoped_store_keys(keys),
            shard_indices,
            ptrs,
        )
        return FAWALoadTask(
            request_id=request_id,
            label=label,
            store=store,
            task=task,
            key_count=len(keys),
            anchor_vllm_block_ids=anchor_vllm_block_ids,
        )

    def _submit_dump_task(
        self,
        label: str,
        store,
        keys: list[bytes],
        ptrs: np.ndarray,
        event_handle,
    ) -> FAWADumpTask:
        return super()._submit_dump_task(
            label,
            store,
            self._rank_scoped_store_keys(keys),
            ptrs,
            event_handle,
        )

    def _submit_dump_task_with_shards(
        self,
        label: str,
        store,
        keys: list[bytes],
        shard_indices: list[int],
        ptrs: np.ndarray,
        event_handle,
    ) -> FAWADumpTask:
        task = store.dump_data(
            self._rank_scoped_store_keys(keys),
            shard_indices,
            ptrs,
            event_handle,
        )
        return FAWADumpTask(
            label=label,
            store=store,
            task=task,
            key_count=len(keys),
            event_handle=event_handle,
        )

    def _wait_dump_task(self, dump_task: FAWADumpTask) -> None:
        try:
            dump_task.store.wait(dump_task.task)
        except Exception as e:
            logger.error(
                "Hybrid FAWA dump task failed; external cache may miss. "
                f"label={dump_task.label}, keys={dump_task.key_count}, "
                f"{type(e).__name__}: {e}"
            )
        finally:
            self.device.destroy_event_handle(dump_task.event_handle)

    def start_load_kv(self, forward_context, **kwargs) -> None:
        if not getattr(self, "_wa_state_row_sharded", False):
            return super().start_load_kv(forward_context, **kwargs)

        metadata = self._get_connector_metadata()
        if not isinstance(metadata, UCMFAWAConnectorMetadata):
            raise RuntimeError(f"Unexpected FAWA metadata type: {type(metadata)}")

        tasks: list[FAWALoadTask] = []
        for request_id, request in metadata.request_meta.items():
            if not request.load_keys:
                continue
            group0_vllm_block_ids = set(request.load_vllm_block_ids[0])
            try:
                if self.fa_store is None:
                    raise RuntimeError("FA store is not initialized.")
                if self.wa_store is None:
                    raise RuntimeError("WA store is not initialized.")

                fa_ptrs = self._extract_fa_ptr(
                    request.load_keys,
                    request.load_hash_start,
                    request.load_hash_end,
                    request.load_vllm_block_ids,
                )
                tasks.append(
                    self._submit_load_task(
                        request_id,
                        "FA",
                        self.fa_store,
                        request.load_keys,
                        fa_ptrs,
                        group0_vllm_block_ids,
                    )
                )

                window_keys = request.load_keys[-1:]
                wa_keys, wa_shards, wa_ptrs = self._build_wa_state_row_task_data(
                    window_keys,
                    request.load_vllm_block_ids,
                )
                tasks.append(
                    self._submit_load_task_with_shards(
                        request_id,
                        "WA",
                        self.wa_store,
                        wa_keys,
                        wa_shards,
                        wa_ptrs,
                        group0_vllm_block_ids,
                    )
                )
            except Exception as e:
                logger.error(
                    f"request {request_id} submit hybrid FAWA load task "
                    f"error. {type(e).__name__}: {e}"
                )
                self._invalid_block_ids.update(group0_vllm_block_ids)

        for load_task in tasks:
            self._wait_load_task(load_task)

    def wait_for_save(self) -> None:
        """Persist every TP rank under rank-scoped keys.

        The base FAWA connector balances canonical blocks across TP ranks.
        Qwen hybrid KV/state tensors are TP-sharded, so every rank must save
        and later load its own shard for every reusable block.
        """

        metadata = self._get_connector_metadata()
        if not isinstance(metadata, UCMFAWAConnectorMetadata):
            raise RuntimeError(f"Unexpected FAWA metadata type: {type(metadata)}")

        if self.fa_store is None:
            raise RuntimeError("FA store is not initialized.")
        if self.wa_store is None:
            raise RuntimeError("WA store is not initialized.")

        fa_dump_keys: list[bytes] = []
        wa_dump_keys: list[bytes] = []
        fa_ptr_rows: list[np.ndarray] = []
        wa_ptr_rows: list[np.ndarray] = []
        wa_sharded_keys: list[bytes] = []
        wa_shard_indices: list[int] = []
        wa_sharded_ptr_rows: list[np.ndarray] = []
        dump_tasks: list[FAWADumpTask] = []

        for request_id, request in metadata.request_meta.items():
            if not request.dump_keys:
                continue
            fa_dump_keys.extend(request.dump_keys)
            fa_ptr_rows.append(
                self._extract_fa_ptr(
                    request.dump_keys,
                    request.dump_hash_start,
                    request.dump_hash_end,
                    request.dump_vllm_block_ids,
                )
            )

            if getattr(self, "_wa_state_row_sharded", False):
                task_keys, shard_indices, ptrs = self._build_wa_state_row_task_data(
                    request.dump_keys,
                    request.dump_vllm_block_ids,
                )
                wa_sharded_keys.extend(task_keys)
                wa_shard_indices.extend(shard_indices)
                wa_sharded_ptr_rows.append(ptrs)
            else:
                wa_dump_keys.extend(request.dump_keys[-1:])
                wa_ptr_rows.append(
                    self._extract_wa_ptr(
                        request.dump_keys[-1:],
                        request.dump_vllm_block_ids,
                    )
                )

        if fa_dump_keys:
            event_handle = self._get_dump_event_handle()
            try:
                dump_tasks.append(
                    self._submit_dump_task(
                        "FA",
                        self.fa_store,
                        fa_dump_keys,
                        np.vstack(fa_ptr_rows),
                        event_handle,
                    )
                )
            except Exception as e:
                self.device.destroy_event_handle(event_handle)
                logger.error(
                    f"dump hybrid FAWA FA kv cache failed. {type(e).__name__}: {e}"
                )
        if wa_sharded_keys:
            event_handle = self._get_dump_event_handle()
            try:
                dump_tasks.append(
                    self._submit_dump_task_with_shards(
                        "WA",
                        self.wa_store,
                        wa_sharded_keys,
                        wa_shard_indices,
                        np.vstack(wa_sharded_ptr_rows),
                        event_handle,
                    )
                )
            except Exception as e:
                self.device.destroy_event_handle(event_handle)
                logger.error(
                    f"dump hybrid FAWA WA kv cache failed. {type(e).__name__}: {e}"
                )
        elif wa_dump_keys:
            event_handle = self._get_dump_event_handle()
            try:
                dump_tasks.append(
                    self._submit_dump_task(
                        "WA",
                        self.wa_store,
                        wa_dump_keys,
                        np.vstack(wa_ptr_rows),
                        event_handle,
                    )
                )
            except Exception as e:
                self.device.destroy_event_handle(event_handle)
                logger.error(
                    f"dump hybrid FAWA WA kv cache failed. {type(e).__name__}: {e}"
                )

        for dump_task in dump_tasks:
            self._wait_dump_task(dump_task)

    def _drain_best_effort_dump_tasks(self, finished_req_ids: set[str]) -> None:
        if not finished_req_ids:
            return

        finished_chunk_req_ids = []
        for request_ids, dump_tasks in self.tp_dump_tasks.items():
            if finished_req_ids.intersection(request_ids):
                finished_chunk_req_ids.append(request_ids)
                for dump_task in dump_tasks:
                    self._wait_dump_task(dump_task)

        for request_ids in finished_chunk_req_ids:
            self.tp_dump_tasks.pop(request_ids, None)

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        self._drain_best_effort_dump_tasks(finished_req_ids)
        return None, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, object] | None]:
        return False, None

    def _generate_dispatch_meta(
        self,
        req_meta: FAWARequestMeta,
        new_tokens: int,
        new_vllm_block_ids: tuple[list[int], ...],
        need_load: bool = True,
    ) -> FAWARequestDispatchMeta:
        if not req_meta.vllm_block_ids:
            req_meta.vllm_block_ids = tuple(
                [] for _ in self._kv_cache_config.kv_cache_groups
            )
        if len(new_vllm_block_ids) != len(req_meta.vllm_block_ids):
            raise RuntimeError(
                f"Hybrid FAWA dispatch metadata expected "
                f"{len(req_meta.vllm_block_ids)} KV cache groups, "
                f"got {len(new_vllm_block_ids)}."
            )
        for group_id, block_ids in enumerate(new_vllm_block_ids):
            req_meta.vllm_block_ids[group_id].extend(block_ids)

        all_group_block_ids = req_meta.vllm_block_ids
        load_block_keys: list[bytes] = []
        load_start, load_end = 0, 0
        load_vllm_block_ids: list[list[int]] = []
        if need_load and req_meta.total_hit_block_num > req_meta.hbm_hit_block_num:
            load_start = req_meta.hbm_hit_block_num
            load_end = req_meta.total_hit_block_num
            load_block_keys = req_meta.ucm_block_ids[load_start:load_end]
            window_boundary_token_idx = (
                np.arange(load_start + 1, load_end + 1) * self.hash_block_size - 1
            )
            for group_id, group_block_ids in enumerate(all_group_block_ids):
                load_vllm_block_ids.append(
                    self._slice_group_block_ids_for_reason(
                        group_id,
                        group_block_ids,
                        window_boundary_token_idx,
                        "load",
                    )
                )

        computed_end_token = min(
            req_meta.num_token_ids,
            req_meta.token_processed + new_tokens,
        )
        dump_start = req_meta.token_processed // self.hash_block_size
        dump_end = computed_end_token // self.hash_block_size
        dump_block_keys: list[bytes] = []
        dump_vllm_block_ids: list[list[int]] = []
        if dump_end > dump_start:
            dump_block_keys = req_meta.ucm_block_ids[dump_start:dump_end]
            window_boundary_token_idx = (
                np.arange(dump_start + 1, dump_end + 1) * self.hash_block_size - 1
            )
            for group_id, group_block_ids in enumerate(all_group_block_ids):
                dump_vllm_block_ids.append(
                    self._slice_group_block_ids_for_reason(
                        group_id,
                        group_block_ids,
                        window_boundary_token_idx,
                        "dump",
                    )
                )
        req_meta.token_processed = computed_end_token

        return FAWARequestDispatchMeta(
            load_keys=load_block_keys,
            load_hash_start=load_start,
            load_hash_end=load_end,
            load_vllm_block_ids=tuple(load_vllm_block_ids),
            dump_keys=dump_block_keys,
            dump_hash_start=dump_start,
            dump_hash_end=dump_end,
            dump_vllm_block_ids=tuple(dump_vllm_block_ids),
        )


def _sample_spec(spec: "KVCacheSpec") -> "KVCacheSpec":
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return next(iter(spec.kv_cache_specs.values()))
    return spec


def _block_size_from_spec(spec: "KVCacheSpec") -> int:
    return int(_sample_spec(spec).block_size)
