import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch
from vllm.distributed.utils import get_pp_indices
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

from ucm.logger import init_logger
from ucm.utils import Config

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class KVCacheType(Enum):
    ATTENTION = "attention"
    MAMBA = "mamba"
    UNKNOWN = "unknown"


@dataclass
class GroupLayout:
    """Per-group layout: identity, physical pointers, and store slot mapping."""

    group_id: int
    group_kind: KVCacheType
    layer_ids: list[int] = field(default_factory=list)
    layer_id_to_row: dict[int, int] = field(default_factory=dict)
    kernel_block_scale: int = 1

    base_ptrs: Optional[np.ndarray] = None
    strides: Optional[np.ndarray] = None
    tensor_sizes: Optional[np.ndarray] = None

    slot_mapping: Optional[list[int]] = None
    num_store_slots: int = 0

    group_spec: Optional[KVCacheGroupSpec] = None

    @property
    def num_layers(self) -> int:
        return len(self.layer_ids)

    @property
    def num_tensors(self) -> int:
        if self.base_ptrs is None:
            return 0
        return self.base_ptrs.shape[1]

    def resolve(self, block_ids: List[int], layer_first: bool = False) -> np.ndarray:
        block_ids_np = np.asarray(block_ids, dtype=np.uint64)
        if layer_first:
            return (
                block_ids_np[None, :, None] * self.strides[:, None, :]
                + self.base_ptrs[:, None, :]
            )
        return (
            block_ids_np[:, None, None] * self.strides[None, :, :]
            + self.base_ptrs[None, :, :]
        )


class KVCacheLayout:
    """KV cache layout for non-hybrid models"""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        ucm_config: Config,
        kv_cache_config: Optional[KVCacheConfig] = None,
    ) -> None:
        self.use_layerwise = ucm_config.get_config().get("use_layerwise", False)
        self.kv_cache_config = kv_cache_config
        self.vllm_config = vllm_config
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.num_hidden_layers = getattr(
            vllm_config.model_config.hf_text_config, "num_hidden_layers", 0
        )
        pp_rank = (
            vllm_config.parallel_config.rank
            // vllm_config.parallel_config.tensor_parallel_size
        ) % self.pp_size
        start, end = get_pp_indices(self.num_hidden_layers, pp_rank, self.pp_size)
        self.local_num_hidden_layers = end - start
        if self.pp_size > 1 and self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be > 0 when pp_size > 1")
        self.cache_block_size = vllm_config.cache_config.block_size

        self.groups: dict[int, GroupLayout] = {}
        self.group_ids_by_kind: dict[KVCacheType, list[int]] = defaultdict(list)
        self.layer_name_to_id: dict[str, int] = {}
        self.layer_name_to_group_id: dict[str, int] = {}
        self.first_layer_id: int = 0
        self.layer_ids: list[int] = []

        self._tensor_size_list: Optional[list[int]] = None
        self._shard_size: Optional[int] = None
        self._block_size: Optional[int] = None
        self._num_layers_per_group: int = 0

        self._init_from_spec()

    def _init_from_spec(self):
        kv_cache_groups = (
            self.kv_cache_config.kv_cache_groups
            if self.kv_cache_config is not None
            and getattr(self.kv_cache_config, "kv_cache_groups", None)
            else []
        )
        if not kv_cache_groups:
            self.groups[0] = GroupLayout(group_id=0, group_kind=KVCacheType.ATTENTION)
            self.group_ids_by_kind[KVCacheType.ATTENTION].append(0)

        for gid, group_spec in enumerate(kv_cache_groups):
            kind = _spec_to_kind(group_spec.kv_cache_spec)
            self.groups[gid] = GroupLayout(
                group_id=gid, group_kind=kind, group_spec=group_spec
            )
            self.group_ids_by_kind[kind].append(gid)
            for layer_name in group_spec.layer_names:
                self.layer_name_to_group_id[layer_name] = gid

    def initialize_kv_cache_layout(self, kvcaches):
        if not self.kv_cache_config:
            for layer_name in kvcaches.keys():
                self.layer_name_to_group_id[layer_name] = 0
        self._register_pointers(kvcaches)
        self._num_layers_per_group = self.groups[self.default_group_id].num_layers
        for group_id in self.groups.keys():
            if self.groups[group_id].num_layers != self._num_layers_per_group:
                raise ValueError(
                    f"All groups must have the same number of layers, but group {group_id} has {self.groups[group_id].num_layers} layers"
                )
        self.layer_ids = sorted(set(self.layer_name_to_id.values()))
        self.first_layer_id = self.layer_ids[0]

    @property
    def use_attn_mamba_hybrid(self) -> bool:
        return (
            KVCacheType.ATTENTION in self.group_ids_by_kind
            and KVCacheType.MAMBA in self.group_ids_by_kind
        )

    @property
    def kv_cache_types(self) -> list[KVCacheType]:
        return list(self.group_ids_by_kind.keys()) or [KVCacheType.ATTENTION]

    @property
    def default_kv_cache_type(self) -> KVCacheType:
        if KVCacheType.ATTENTION in self.group_ids_by_kind:
            return KVCacheType.ATTENTION
        return self.kv_cache_types[0]

    @property
    def default_group_id(self) -> int:
        attn_ids = self.group_ids_by_kind.get(KVCacheType.ATTENTION, [])
        return attn_ids[0] if attn_ids else next(iter(self.groups))

    @property
    def group_ids_by_kv_cache_type(self) -> dict[KVCacheType, list[int]]:
        return self.group_ids_by_kind

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    def extract_block_addrs(
        self,
        vllm_block_ids: List[int],
        group_id: int,
        layer_first: bool = False,
    ) -> np.ndarray:
        return self.groups[group_id].resolve(vllm_block_ids, layer_first)

    def tensor_size_list(self) -> list[int]:
        if self._tensor_size_list is not None:
            return self._tensor_size_list
        g = self.groups[self.default_group_id]
        if self.use_layerwise:
            return g.tensor_sizes[0].tolist()
        return g.tensor_sizes.reshape(-1).tolist()

    def shard_size(self) -> int:
        if self._shard_size is not None:
            return self._shard_size
        g = self.groups[self.default_group_id]
        if self.use_layerwise:
            return int(g.tensor_sizes[0].sum())
        return int(g.tensor_sizes.sum())

    def block_size(self) -> int:
        if self._block_size is not None:
            return self._block_size
        g = self.groups[self.default_group_id]
        if self.pp_size > 1:
            return int(g.tensor_sizes[0].sum() * self.num_hidden_layers)
        return int(g.tensor_sizes.sum())

    def _register_pointers(self, kvcaches):
        raw_ptr_rows: dict[int, list[list[int]]] = defaultdict(list)
        stride_rows: dict[int, list[list[int]]] = defaultdict(list)
        tensor_size_rows: dict[int, list[list[int]]] = defaultdict(list)
        layer_ids_by_group: dict[int, list[int]] = defaultdict(list)
        layer_id_to_row_by_group: dict[int, dict[int, int]] = defaultdict(dict)
        kernel_block_scale_by_group: dict[int, int] = {}

        for layer_name, kv_layer in kvcaches.items():
            group_id = self.layer_name_to_group_id.get(layer_name, 0)
            layer_id = extract_layer_index(layer_name)
            self.layer_name_to_id[layer_name] = layer_id
            ptrs, strides, tensor_sizes, kbs = self._handle_kv_layer(kv_layer, group_id)

            row_idx = layer_id_to_row_by_group[group_id].get(layer_id)
            if row_idx is None:
                row_idx = len(layer_ids_by_group[group_id])
                layer_id_to_row_by_group[group_id][layer_id] = row_idx
                layer_ids_by_group[group_id].append(layer_id)
                raw_ptr_rows[group_id].append(list(ptrs))
                stride_rows[group_id].append(list(strides))
                tensor_size_rows[group_id].append(list(tensor_sizes))
            else:
                raw_ptr_rows[group_id][row_idx].extend(ptrs)
                stride_rows[group_id][row_idx].extend(strides)
                tensor_size_rows[group_id][row_idx].extend(tensor_sizes)

            kernel_block_scale_by_group[group_id] = kbs

        for group_id, layer_ids in layer_ids_by_group.items():
            g = self.groups[group_id]
            g.layer_ids = layer_ids
            g.layer_id_to_row = {lid: idx for idx, lid in enumerate(layer_ids)}
            g.kernel_block_scale = kernel_block_scale_by_group.get(group_id, 1)
            g.base_ptrs = np.asarray(raw_ptr_rows[group_id], dtype=np.uint64)
            g.strides = np.asarray(stride_rows[group_id], dtype=np.uint64)
            g.tensor_sizes = np.asarray(tensor_size_rows[group_id], dtype=np.uint64)

        self._finalize_store_slot_layout()

        for gid, g in self.groups.items():
            logger.info(
                f"group[{gid}] ({g.group_kind.value}): "
                f"layers={g.num_layers}, tensors={g.num_tensors}, "
                f"shard_size={int(g.tensor_sizes[0].sum())}"
            )

    def _finalize_store_slot_layout(self) -> None:
        """Widen last dim to unified store slots; padded columns are stride/base/size 0."""
        for g in self.groups.values():
            if g.slot_mapping is None or g.num_store_slots == 0:
                continue
            n_layers, n_src = g.base_ptrs.shape
            n_slot = g.num_store_slots
            new_base = np.zeros((n_layers, n_slot), dtype=np.uint64)
            new_strides = np.zeros((n_layers, n_slot), dtype=np.uint64)
            new_sizes = np.zeros((n_layers, n_slot), dtype=np.uint64)
            for src_i, dst in enumerate(g.slot_mapping):
                new_base[:, dst] = g.base_ptrs[:, src_i]
                new_strides[:, dst] = g.strides[:, src_i]
                new_sizes[:, dst] = g.tensor_sizes[:, src_i]
            g.base_ptrs = new_base
            g.strides = new_strides
            g.tensor_sizes = new_sizes
            g.slot_mapping = None
            g.num_store_slots = 0

    def _handle_kv_layer(self, kv_layer, group_id: int):
        ptrs: list[int] = []
        strides: list[int] = []
        tensor_sizes: list[int] = []
        kernel_block_scale = 1

        def handle_kv_tensor(t: torch.Tensor):
            nonlocal kernel_block_scale
            ptrs.append(t.data_ptr())
            strides.append(t.stride(0) * t.element_size())
            ts = math.prod(t.shape[i] for i in range(1, t.dim())) * t.element_size()
            spec = self.groups[group_id].group_spec.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = next(iter(spec.kv_cache_specs.values()))
            if spec is not None and isinstance(spec, AttentionSpec):
                kernel_block_size = t.shape[1]
                kernel_block_scale = self.cache_block_size // kernel_block_size
                ts *= kernel_block_scale
            tensor_sizes.append(ts)

        if isinstance(kv_layer, torch.Tensor):
            if kv_layer.dim() == 5 and kv_layer.shape[0] == 2:
                handle_kv_tensor(kv_layer[0])
                handle_kv_tensor(kv_layer[1])
            else:
                handle_kv_tensor(kv_layer)
        elif isinstance(kv_layer, (tuple, list)):
            for tensor in kv_layer:
                handle_kv_tensor(tensor)
        else:
            raise TypeError(f"Unsupported kv cache type: {type(kv_layer)}")

        return ptrs, strides, tensor_sizes, kernel_block_scale


class CudaHybridLayout(KVCacheLayout):
    """KV cache layout for CUDA mamba hybrid models, e.g. Qwen3-next"""

    def _init_from_spec(self):
        super()._init_from_spec()
        attn_gids = self.group_ids_by_kind.get(KVCacheType.ATTENTION, [])

        if not attn_gids:
            raise ValueError(
                "CudaHybridLayout requires both ATTENTION and MAMBA groups"
            )
        attn_group_spec = self.groups[attn_gids[0]].group_spec
        page_size = attn_group_spec.kv_cache_spec.page_size_bytes

        for g in self.groups.values():
            g.slot_mapping = [0]
            g.num_store_slots = 1

        self._tensor_size_list = [page_size] * (1 if self.use_layerwise else len(attn_group_spec.layer_names))
        self._shard_size = sum(self._tensor_size_list)
        self._block_size = page_size * len(attn_group_spec.layer_names)

        logger.info(
            f"CudaHybridLayout: page_size={page_size}, num_layers={len(attn_group_spec.layer_names)}, "
            f"slot_mapping=[0] for all groups"
        )


class AscendHybridLayout(KVCacheLayout):
    """KV cache layout for Ascend mamba hybrid models, e.g. Qwen3-next"""

    def _init_from_spec(self):
        super()._init_from_spec()
        attn_gids = self.group_ids_by_kind.get(KVCacheType.ATTENTION, [])
        mamba_gids = self.group_ids_by_kind.get(KVCacheType.MAMBA, [])

        if not attn_gids or not mamba_gids:
            raise ValueError(
                "AscendHybridLayout requires both ATTENTION and MAMBA groups"
            )

        attn_group_spec = self.groups[attn_gids[0]].group_spec
        mamba_group_spec = self.groups[mamba_gids[0]].group_spec
        assert attn_group_spec is not None and mamba_group_spec is not None

        conv_size = math.prod(
            mamba_group_spec.kv_cache_spec.shapes[0]
        ) * get_dtype_size(mamba_group_spec.kv_cache_spec.dtypes[0])
        k_size = (
            attn_group_spec.kv_cache_spec.block_size
            * attn_group_spec.kv_cache_spec.num_kv_heads
            * attn_group_spec.kv_cache_spec.head_size
            * get_dtype_size(attn_group_spec.kv_cache_spec.dtype)
        )
        v_size = k_size

        unified_num_slots = 3
        for gid in attn_gids:
            self.groups[gid].slot_mapping = [1, 2]
            self.groups[gid].num_store_slots = unified_num_slots
        for gid in mamba_gids:
            self.groups[gid].slot_mapping = [0, 1]
            self.groups[gid].num_store_slots = unified_num_slots

        self._tensor_size_list = [conv_size, k_size, v_size] * (1 if self.use_layerwise else len(attn_group_spec.layer_names))
        self._shard_size = sum(self._tensor_size_list)
        self._block_size = sum(self._tensor_size_list) * len(attn_group_spec.layer_names)

        for gid, g in self.groups.items():
            logger.info(
                f"AscendHybridLayout group[{gid}] ({g.group_kind.value}): "
                f"slot_mapping={g.slot_mapping}"
            )
        logger.info(
            f"AscendHybridLayout: tensor_size_list={self._tensor_size_list}, "
            f"shard_size={self._shard_size}"
        )


def _spec_to_kind(spec) -> KVCacheType:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        assert spec.is_uniform_type(
            spec.kv_cache_specs
        ), "UniformTypeKVCacheSpecs must be uniform type"
        spec = next(iter(spec.kv_cache_specs.values()))
    if isinstance(spec, MambaSpec):
        return KVCacheType.MAMBA
    if isinstance(spec, AttentionSpec):
        return KVCacheType.ATTENTION
    return KVCacheType.UNKNOWN


def _check_is_attn_mamba_hybrid(kv_cache_config: Optional[KVCacheConfig]) -> bool:
    if kv_cache_config is None:
        return False
    groups = getattr(kv_cache_config, "kv_cache_groups", None)
    if not groups:
        return False
    use_attn, use_mamba = False, False
    for group_spec in groups:
        if _spec_to_kind(group_spec.kv_cache_spec) == KVCacheType.MAMBA:
            use_mamba = True
        if _spec_to_kind(group_spec.kv_cache_spec) == KVCacheType.ATTENTION:
            use_attn = True
    return use_attn and use_mamba


def create_kv_cache_layout(
    vllm_config: "VllmConfig",
    ucm_config: Config,
    kv_cache_config: Optional[KVCacheConfig] = None,
) -> KVCacheLayout:
    use_attn_mamba_hybrid = _check_is_attn_mamba_hybrid(kv_cache_config)

    if not use_attn_mamba_hybrid:
        return KVCacheLayout(vllm_config, ucm_config, kv_cache_config)

    if current_platform.is_cuda_alike():
        return CudaHybridLayout(vllm_config, ucm_config, kv_cache_config)

    if current_platform.device_type == "npu":
        return AscendHybridLayout(vllm_config, ucm_config, kv_cache_config)

    raise RuntimeError(
        f"Unsupported platform for hybrid model: {current_platform.device_type}"
    )
