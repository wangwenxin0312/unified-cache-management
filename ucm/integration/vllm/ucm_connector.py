import copy
import hashlib
import math
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import get_world_group
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput

from ucm.integration.vllm.device import create_device
from ucm.logger import init_logger
from ucm.observability import PrometheusStatsLogger
from ucm.shared.metrics import ucmmetrics
from ucm.store.factory_v1 import UcmConnectorFactoryV1
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1
from ucm.utils import Config

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

from ucm.sparse.state import has_ucm_sparse

try:
    from vllm.v1.core.kv_cache_utils import KVCacheConfig, MambaSpec
    _HAS_KV_CACHE_CONFIG = True
except ImportError:
    _HAS_KV_CACHE_CONFIG = False

logger = init_logger(__name__)


@dataclass
class RequestMeta:
    ucm_block_ids: list[bytes] = field(default_factory=list)
    hbm_hit_block_num: int = 0
    # local_computed_block + external_computed_block
    total_hit_block_num: int = 0
    num_token_ids: int = 0
    vllm_block_ids: list[int] = field(default_factory=list)
    token_processed: int = 0
    # Per mamba group: mamba_vllm_ids[k] is the physical block list for group k
    mamba_vllm_ids: list[list[int]] = field(default_factory=list)


@dataclass
class RequestDispatchMeta:
    load_block_ids: tuple[
        list[bytes], list[int]
    ]  # [0] mean ucm_block_ids, [1] means vllm_block_ids
    dump_block_ids: tuple[list[bytes], list[int]]
    # Per mamba group: (ucm_ids, vllm_ids) for load and dump
    mamba_load: list[tuple[list[bytes], list[int]]] = field(default_factory=list)
    mamba_dump: list[tuple[list[bytes], list[int]]] = field(default_factory=list)


class KVCacheLayout:
    def __init__(
        self, kvcaches, use_layerwise: bool, vllm_config: "VllmConfig"
    ) -> None:
        # each row is a layer, each column is a tensor_size/ptr in the layer (e.g., k, v, rope, k_index)
        self.base_ptrs: np.ndarray  # (n_layers, n_ptrs）
        self.block_stride_lists: np.ndarray  # (n_layers, n_tensor_sizes)
        self.tensor_size_lists: np.ndarray  # (n_layers, n_tensor_sizes)
        self.use_layerwise = use_layerwise
        self.vllm_config = vllm_config
        self.pp_size = self.vllm_config.parallel_config.pipeline_parallel_size
        self.num_hidden_layers = getattr(
            self.vllm_config.model_config.hf_text_config, "num_hidden_layers", 0
        )
        if self.pp_size > 1 and self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be > 0 when pp_size > 1")
        self.layer_name_to_id = {
            name: extract_layer_index(name) for name in kvcaches.keys()
        }
        self.layer_ids = sorted(set(self.layer_name_to_id.values()))
        if not self.layer_ids:
            raise ValueError("KV cache layout requires at least one layer.")
        self.layer_id_to_row = {
            layer_id: row for row, layer_id in enumerate(self.layer_ids)
        }
        self.first_layer_id = self.layer_ids[0]
        self._build_layout(kvcaches)

    def _build_layout(self, kvcaches):

        num_rows = len(self.layer_ids)
        raw_ptr_rows = [[] for _ in range(num_rows)]
        block_stride_rows = [[] for _ in range(num_rows)]
        tensor_size_rows = [[] for _ in range(num_rows)]

        for layer_name, kv_layer in kvcaches.items():
            ptrs = []
            block_strides = []
            tensor_sizes = []

            def handle_tensor(t: torch.Tensor, size_dims):
                ptrs.append(t[0].data_ptr())

                tensor_size = (
                    math.prod([t.shape[i] for i in size_dims]) * t.element_size()
                )
                tensor_sizes.append(tensor_size)
                block_strides.append(t.stride(0) * t.element_size())

            def handle_block_tensor(t: torch.Tensor):
                if t.dim() < 1:
                    raise ValueError(
                        f"Unsupported kv cache tensor shape: {t.shape}"
                    )
                ptrs.append(t[0].data_ptr())
                tensor_sizes.append(math.prod(t.shape[1:]) * t.element_size())
                block_strides.append(t.stride(0) * t.element_size())

            if isinstance(kv_layer, torch.Tensor):
                if kv_layer.dim() == 5:
                    # [2, num_blocks, block_size, num_head, head_dim]
                    handle_tensor(kv_layer[0], (-3, -2, -1))
                    handle_tensor(kv_layer[1], (-3, -2, -1))
                elif kv_layer.dim() == 3:
                    # [num_blocks, block_size, head_dim]
                    handle_block_tensor(kv_layer)
                else:
                    raise ValueError(
                        f"Unsupported kv cache tensor shape: {kv_layer.shape}"
                    )
            elif isinstance(kv_layer, Sequence):
                # vllm_ascend >= 0.10.0 uses a tuple for attention caches:
                # ([num_blocks, block_size, num_head, head_dim], ...).
                # Mamba/GDN layers use a list of state tensors, e.g.
                # conv_state [num_blocks, conv_dim, conv_width] and
                # ssm_state [num_blocks, heads, state_dim, head_dim].
                for tensor in kv_layer:
                    handle_block_tensor(tensor)
            else:
                raise TypeError(f"Unsupported kv cache type: {type(kv_layer)}")

            local_layer_id = self.layer_id_to_row[self.layer_name_to_id[layer_name]]
            raw_ptr_rows[local_layer_id].extend(ptrs)
            block_stride_rows[local_layer_id].extend(block_strides)
            tensor_size_rows[local_layer_id].extend(tensor_sizes)

        row_widths = {len(row) for row in raw_ptr_rows}
        if len(row_widths) != 1:
            raise ValueError(
                "Unsupported mixed KV cache layout with different tensor counts "
                f"per layer: {sorted(row_widths)}"
            )

        self.base_ptrs = np.asarray(raw_ptr_rows, dtype=np.uint64)
        self.block_stride_lists = np.asarray(block_stride_rows, dtype=np.uint64)
        self.tensor_size_lists = np.asarray(tensor_size_rows, dtype=np.uint64)

        logger.info(
            f"KVCacheLayout built: "
            f"base_ptrs shape={self.base_ptrs.shape}, "
            f"block_stride_lists shape={self.block_stride_lists.shape}, "
            f"tensor_size_lists shape={self.tensor_size_lists.shape}"
        )
        logger.info(
            f"Logical tensor sizes  (tensor_size_lists): "
            f"{self.tensor_size_lists.reshape(-1).tolist()}"
        )
        logger.info(
            f"Physical block strides (block_stride_lists): "
            f"{self.block_stride_lists.reshape(-1).tolist()}"
        )

    def extract_block_addrs(
        self,
        vllm_block_ids: List[int],
        mamba_vllm_block_ids: Optional[list[list[int]]] = None,
        layer_first: bool = False,
    ) -> np.ndarray:
        vllm_block_ids_np = np.array(vllm_block_ids, np.uint64)
        if layer_first:
            # (n_layers, num_blocks, n_ptrs)
            return (
                self.block_stride_lists[:, None, :]
                * vllm_block_ids_np[None, :, None]
                + self.base_ptrs[:, None, :]
            )
        return (
            vllm_block_ids_np[:, None, None] * self.block_stride_lists[None, :, :]
            + self.base_ptrs[None, :, :]
        )  # (num_blocks, n_layers, n_ptrs)

    @property
    def tensor_size_list(self) -> list[int]:
        # Use block_stride_lists (physical stride per block) rather than
        # tensor_size_lists (logical content size). On Ascend NPU, tensors are
        # allocated with 262144-byte alignment so stride(0)*element_size equals
        # the aligned size (e.g. 262144) for every tensor, even when the logical
        # content is only 12288 or 65536 bytes. On CUDA with contiguous tensors,
        # stride == logical size, so behaviour is unchanged.
        if self.use_layerwise and not np.all(
            self.block_stride_lists == self.block_stride_lists[0]
        ):
            raise RuntimeError(
                "Layerwise UCM currently requires identical tensor sizes for "
                "every layer. Hybrid attention/Mamba models such as Qwen3-Next "
                "should use use_layerwise=False so attention blocks and "
                "conv/ssm Mamba blocks are dumped together."
            )
        result = (
            self.block_stride_lists.reshape(-1).tolist()
            if not self.use_layerwise
            else self.block_stride_lists[0].tolist()
        )
        logger.info(
            f"tensor_size_list (physical strides): {result}"
        )
        return result

    @property
    def shard_size(self) -> int:
        return int(
            self.block_stride_lists.sum()
            if not self.use_layerwise
            else self.block_stride_lists[0].sum()
        )

    @property
    def block_size(self) -> int:
        if self.pp_size > 1:
            return int(self.block_stride_lists[0].sum() * self.num_hidden_layers)
        return int(self.block_stride_lists.sum())


class HybridKVCacheLayout:
    """Single-store layout for attention + Mamba cache blocks.

    Column order is attention K/V for all attention layers, then each Mamba
    group's conv/ssm tensors. tensor_size_list describes how many bytes to copy;
    each child KVCacheLayout still owns the correct per-block stride.
    """

    def __init__(
        self,
        attn_layout: KVCacheLayout,
        mamba_layouts: list[Optional[KVCacheLayout]],
    ) -> None:
        self.attn_layout = attn_layout
        self.mamba_layouts = mamba_layouts
        self.use_layerwise = False
        self.vllm_config = attn_layout.vllm_config
        self.layer_name_to_id = attn_layout.layer_name_to_id
        self.layer_ids = attn_layout.layer_ids
        self.layer_id_to_row = attn_layout.layer_id_to_row
        self.first_layer_id = attn_layout.first_layer_id

    @property
    def tensor_size_list(self) -> list[int]:
        ret = self.attn_layout.tensor_size_list
        for layout in self.mamba_layouts:
            if layout is not None:
                ret.extend(layout.tensor_size_list)
        return ret

    @property
    def shard_size(self) -> int:
        return int(sum(self.tensor_size_list))

    @property
    def block_size(self) -> int:
        return self.shard_size

    def extract_block_addrs(
        self,
        vllm_block_ids: List[int],
        mamba_vllm_block_ids: Optional[list[list[int]]] = None,
        layer_first: bool = False,
    ) -> np.ndarray:
        if layer_first:
            raise RuntimeError(
                "Hybrid attention/Mamba layout does not support layerwise "
                "address extraction."
            )

        num_blocks = len(vllm_block_ids)
        parts = [
            self.attn_layout.extract_block_addrs(vllm_block_ids).reshape(
                num_blocks, -1
            )
        ]

        if not any(layout is not None for layout in self.mamba_layouts):
            return parts[0]

        if mamba_vllm_block_ids is None:
            raise RuntimeError(
                "Hybrid attention/Mamba layout requires mamba physical block ids."
            )

        for k, layout in enumerate(self.mamba_layouts):
            if layout is None:
                continue
            if k >= len(mamba_vllm_block_ids):
                raise RuntimeError(f"Missing mamba block ids for group {k}.")
            group_block_ids = mamba_vllm_block_ids[k]
            if len(group_block_ids) != num_blocks:
                raise RuntimeError(
                    f"Mamba group {k} block count {len(group_block_ids)} "
                    f"does not match attention block count {num_blocks}."
                )
            group_ptrs = layout.extract_block_addrs(group_block_ids).reshape(
                num_blocks, -1
            )
            parts.append(group_ptrs)

        return np.concatenate(parts, axis=1)


@dataclass
class UCMConnectorMetadata(KVConnectorMetadata):
    request_meta: dict[str, RequestDispatchMeta] = field(default_factory=dict)


class RequestHasher:
    """hash(md5) request to generate ucm block id"""

    def __init__(self, vllm_config, rank_id):
        meta = f"{vllm_config.model_config.model}:{vllm_config.parallel_config.tensor_parallel_size}:{vllm_config.model_config.dtype}:{rank_id}"
        self.meta_bytes = meta.encode("utf-8")

    def __call__(self, input_data) -> bytes:
        if isinstance(input_data, bytes):
            input_bytes = input_data
        else:
            input_bytes = pickle.dumps(input_data, protocol=pickle.HIGHEST_PROTOCOL)

        h = hashlib.md5(self.meta_bytes + input_bytes)
        return h.digest()


class UCMDirectConnector(KVConnectorBase_V1, SupportsHMA):
    """
    This connector means synchronize:
    load -> forward -> save
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole,
                 kv_cache_config=None):
        super().__init__(vllm_config=vllm_config, role=role)
        self.use_layerwise = False
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.local_rank = (
            -1 if role == KVConnectorRole.SCHEDULER else get_world_group().local_rank
        )
        self.tp_rank = self._vllm_config.parallel_config.rank
        self.block_size = self._vllm_config.cache_config.block_size
        self.is_mla = self._vllm_config.model_config.is_deepseek_mla
        self.num_layers = self._vllm_config.model_config.get_num_layers(
            self._vllm_config.parallel_config
        )
        self.tp_size = self._vllm_config.parallel_config.tensor_parallel_size
        self.kv_cache_dtype: torch.dtype = None
        self.num_head = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.head_size = vllm_config.model_config.get_head_size()
        self.element_size = vllm_config.model_config.dtype.itemsize

        # Mamba / hybrid-model group tracking (populated after kv_cache_config
        # is available, either here or lazily in register_kv_caches).
        # mamba_group_layer_names[k] = sorted list of layer names for mamba group k
        self.mamba_group_layer_names: list[list[str]] = []
        # mamba_block_sizes[k] = token block_size for mamba group k
        self.mamba_block_sizes: list[int] = []
        # Mamba layouts are folded into self.store for hybrid models.
        self.mamba_kv_cache_layouts: list[Optional["KVCacheLayout"]] = []
        self._pending_kv_cache_config = kv_cache_config  # deferred until hasher ready

        if current_platform.is_cuda_alike():
            logger.info("CUDA device is available.")
            torch_dev = torch
            dev_name = "cuda"
        elif current_platform.device_type == "npu":
            logger.info("NPU device is available.")
            torch_dev = torch.npu
            dev_name = "npu"
        else:
            raise RuntimeError("Unsupported device platform for UCMDirectConnector.")

        if self.local_rank >= 0:
            self.device = torch_dev.device(f"{dev_name}:{self.local_rank}")

        self.store: UcmKVStoreBaseV1
        self.rope_store: Optional[UcmKVStoreBaseV1] = None

        # save block info, avoid hash request twice, and track them until request finished
        self.requests_meta: dict[str, RequestMeta] = {}
        self.layer_id_to_row: dict[int, int] = {}

        ucm_config = Config(vllm_config.kv_transfer_config)
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self.launch_config = ucm_config.get_config()
        self.connector_configs = self.launch_config.get("ucm_connectors", [])
        self.enable_event_sync = self.launch_config.get("enable_event_sync", True)
        self.enable_record_traces = self.launch_config.get(
            "enable_record_traces", False
        )
        assert len(self.connector_configs) > 0, "no storage connector name in config."

        self.chunk_size = self.block_size
        self.blocks_per_chunk = self.chunk_size // self.block_size

        if role == KVConnectorRole.SCHEDULER:
            self.request_hasher = RequestHasher(vllm_config, 0)
            self._seed = self.request_hasher("UCM_HASH_SEED")
            # init scheduler-size connector
            self.store = self._create_store(None)
        else:
            self.request_hasher = RequestHasher(
                vllm_config, self.tp_rank % self.tp_size
            )

        # Now that request_hasher is ready, initialise mamba group info.
        if _HAS_KV_CACHE_CONFIG and self._pending_kv_cache_config is not None:
            self._init_mamba_group_info(self._pending_kv_cache_config)
        self._pending_kv_cache_config = None

        self.metrics_config = self.launch_config.get("metrics_config_path", "")
        if self.metrics_config:
            worker_id = (
                f"{self.engine_id}_{get_world_group().rank}"
                if role == KVConnectorRole.WORKER
                else self.engine_id
            )
            self.stats_logger = PrometheusStatsLogger(
                vllm_config.model_config.served_model_name,
                worker_id,
                self.metrics_config,
            )
            logger.info(
                f"metrics_config_path: {self.metrics_config}, set worker_id: {worker_id}"
            )

        self.persist_token_threshold = self.launch_config.get(
            "persist_token_threshold", 0
        )

        # invalid block ids due to load errors
        self._invalid_block_ids: set[int] = set()
        self.cp_world_size = 1
        self.hash_block_size = self.block_size
        self.block_size *= self.cp_world_size

    # ------------------------------------------------------------------
    # Hybrid-model helpers
    # ------------------------------------------------------------------

    def _init_mamba_group_info(self, kv_cache_config) -> None:
        """Populate mamba group metadata from a KVCacheConfig object.

        The first kv_cache_group is treated as the attention group and
        controls the attention block_size already stored in self.block_size.
        All remaining groups that use MambaSpec are registered here.
        """
        if not _HAS_KV_CACHE_CONFIG:
            return
        attn_block_size = self._vllm_config.cache_config.block_size
        for gid, group_spec in enumerate(kv_cache_config.kv_cache_groups):
            spec = group_spec.kv_cache_spec
            if not isinstance(spec, MambaSpec):
                continue
            self.mamba_group_layer_names.append(sorted(group_spec.layer_names))
            self.mamba_block_sizes.append(spec.block_size)
            logger.info(
                f"Registered mamba group {gid}: "
                f"{len(group_spec.layer_names)} layers, "
                f"block_size={spec.block_size}"
            )

    def _validate_single_store_mamba_blocks(self) -> None:
        for k, layout in enumerate(self.mamba_kv_cache_layouts):
            if layout is None:
                continue
            mamba_block_size = self.mamba_block_sizes[k]
            if mamba_block_size != self.hash_block_size:
                raise RuntimeError(
                    "Single-store hybrid UCM requires one attention block to "
                    "map to one Mamba state block. "
                    f"Mamba group {k} block_size={mamba_block_size}, "
                    f"attention block_size={self.hash_block_size}. "
                    "Use mamba_cache_mode='align' or set mamba_block_size to "
                    "the attention block size."
                )

    def _mamba_vllm_ids_from_dispatch(
        self, mamba_dispatch: list[tuple[list[bytes], list[int]]]
    ) -> list[list[int]]:
        return [
            mamba_dispatch[k][1] if k < len(mamba_dispatch) else []
            for k in range(len(self.mamba_kv_cache_layouts))
        ]

    @property
    def num_mamba_groups(self) -> int:
        return len(self.mamba_group_layer_names)

    def generate_hash(
        self, block_size: int, token_ids: List[int], parent_block_hash_value: bytes
    ) -> list[bytes]:
        ret = []
        for start in range(0, len(token_ids), block_size):
            end = start + block_size
            block_token_ids = token_ids[start:end]
            # Do not hash the block if it is not full.
            if len(block_token_ids) < block_size:
                break

            block_token_ids_tuple = tuple(block_token_ids)
            hash_value = self.request_hasher(
                (parent_block_hash_value, block_token_ids_tuple)
            )
            parent_block_hash_value = hash_value
            ret.append(hash_value)

        return ret

    def _create_store(
        self,
        kv_cache_layout: Optional["KVCacheLayout | HybridKVCacheLayout"],
        cpu_affinity_cores: Optional[list[int]] = None,
    ) -> UcmKVStoreBaseV1:
        if len(self.connector_configs) != 1:
            raise RuntimeError(
                f"Expected exactly one connector config, "
                f"but got {len(self.connector_configs)}: "
                f"{self.connector_configs}"
            )

        name = self.connector_configs[0]["ucm_connector_name"]
        module_path = self.connector_configs[0].get("ucm_connector_module_path", None)
        config = copy.deepcopy(self.connector_configs[0]["ucm_connector_config"])
        config.setdefault("share_buffer_enable", self.is_mla)
        if "storage_backends" in config:
            backends = [path for path in config["storage_backends"].split(":")]
            config["storage_backends"] = backends
        config["unique_id"] = f"{self.engine_id}"
        if self._role == KVConnectorRole.WORKER:
            config["device_id"] = self.local_rank
            tensor_size_list = kv_cache_layout.tensor_size_list * self.blocks_per_chunk
            config["tensor_size_list"] = tensor_size_list
            config["shard_size"] = kv_cache_layout.shard_size * self.blocks_per_chunk
            config["block_size"] = kv_cache_layout.block_size * self.blocks_per_chunk
            config["local_rank_size"] = self.tp_size if self.is_mla else 1
            if cpu_affinity_cores:
                config["cpu_affinity_cores"] = list(cpu_affinity_cores)
            logger.info(
                f"Store worker config: "
                f"TensorSizes={tensor_size_list}, "
                f"shard_size={config['shard_size']}, "
                f"block_size={config['block_size']}"
            )
        else:
            config_base = self.block_size * self.element_size * self.head_size
            config["block_size"] = (
                config_base
                * self.num_layers
                * (1 if self.is_mla else self.num_head * 2)
                * self.blocks_per_chunk
            )
        dp_rank = self._vllm_config.parallel_config.data_parallel_rank
        config["posix_gc_enable"] = (
            self._role != KVConnectorRole.WORKER and dp_rank == 0
        )

        logger.info(f"create {name} with config: {config}")
        return UcmConnectorFactoryV1.create_connector(name, config, module_path)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if has_ucm_sparse() and os.getenv("VLLM_HASH_ATTENTION") == "1":
            for layer_name, value in kv_caches.items():
                kv_cache, k_hash = value
                self.kv_caches[layer_name] = kv_cache
        else:
            self.kv_caches = kv_caches
        sample_kv_layer = next(iter(self.kv_caches.values()))
        if self.kv_cache_dtype is None:
            self.kv_cache_dtype = sample_kv_layer[0].dtype
        if isinstance(sample_kv_layer, torch.Tensor):
            logger.info(f"kv cache shape {sample_kv_layer.shape}")
        elif isinstance(sample_kv_layer, Sequence):
            # vllm_ascend >= 0.10.0 uses tuple/list for kvcaches.
            for i, tensor in enumerate(sample_kv_layer):
                logger.info(f"kv cache shape {i}: {tensor.shape}")
        # Build the attention-only sub-dict for the main layout.
        # If no mamba groups were registered (pure attention model), all layers
        # go into the main layout as before.
        all_mamba_layer_names: set[str] = set()
        for names in self.mamba_group_layer_names:
            all_mamba_layer_names.update(names)

        mamba_group_layer_names = self.mamba_group_layer_names
        if all_mamba_layer_names:
            attn_kv_caches = {
                n: v for n, v in self.kv_caches.items()
                if n not in all_mamba_layer_names
            }
            if not attn_kv_caches:
                # Degenerate: all layers are mamba; fall back to combined layout
                attn_kv_caches = self.kv_caches
                mamba_group_layer_names = []
        else:
            attn_kv_caches = self.kv_caches

        attn_layout = KVCacheLayout(
            attn_kv_caches, self.use_layerwise, self._vllm_config
        )
        self.layer_name_to_id = attn_layout.layer_name_to_id
        self.layer_ids = attn_layout.layer_ids
        self.layer_id_to_row = attn_layout.layer_id_to_row
        self.first_layer_id = self.layer_ids[0]

        self.device = create_device()

        enable_affinity = os.getenv("VLLM_CPU_AFFINITY") == "1"
        worker_cores, store_cores = (
            self.device.split_cores(self.local_rank)
            if enable_affinity
            else (None, None)
        )

        # Build mamba layouts and fold them into the main store layout.
        self.mamba_kv_cache_layouts = []
        for k, group_layer_names in enumerate(mamba_group_layer_names):
            mamba_kv = {
                n: self.kv_caches[n]
                for n in group_layer_names
                if n in self.kv_caches
            }
            if not mamba_kv:
                logger.warning(
                    f"Mamba group {k} has no matching kv_cache entries; skipping."
                )
                self.mamba_kv_cache_layouts.append(None)
                continue
            mamba_layout = KVCacheLayout(mamba_kv, False, self._vllm_config)
            self.mamba_kv_cache_layouts.append(mamba_layout)
            logger.info(
                f"Mamba group {k}: layout built for "
                f"{len(mamba_kv)} layers, "
                f"block_size={self.mamba_block_sizes[k]}"
            )

        if any(layout is not None for layout in self.mamba_kv_cache_layouts):
            self._validate_single_store_mamba_blocks()
            self.kv_cache_layout = HybridKVCacheLayout(
                attn_layout, self.mamba_kv_cache_layouts
            )
        else:
            self.kv_cache_layout = attn_layout

        self.block_data_size = self.kv_cache_layout.block_size
        self.store = self._create_store(self.kv_cache_layout, store_cores)

        if worker_cores:
            try:
                os.sched_setaffinity(0, worker_cores)
                logger.info(f"[VLLM CPU Affinity] Worker bound to cores {worker_cores}")
            except Exception as e:
                logger.warning(f"Failed to bind worker: {e}")

        if self.device is None:
            raise RuntimeError(f"Unsupported device platform for UCMDirectConnector.")

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        assert num_computed_tokens % self.block_size == 0
        hbm_hit_block_num = num_computed_tokens // self.block_size

        ucm_block_ids = self.generate_hash(
            self.hash_block_size, request.all_token_ids, self._seed
        )

        if (
            self.enable_record_traces
            and request.request_id not in self.requests_meta
            and len(ucm_block_ids) > 0
        ):
            hex_ucm_block_ids = [id.hex() for id in ucm_block_ids]
            logger.info_once(
                f"timestamp: {time.perf_counter()}, "
                f"input_length: {request.num_tokens}, "
                f"output_length: {request.max_tokens}, "
                f"ucm_block_ids: {hex_ucm_block_ids}"
            )

        # Skip persistence if token count is below the threshold
        if self.persist_token_threshold > request.num_tokens:
            logger.info_once(
                f"Skip persistence: req {request.request_id}, "
                f"input tokens ({request.num_tokens}) < threshold ({self.persist_token_threshold})."
            )
            return 0, False

        external_block_ids = ucm_block_ids[hbm_hit_block_num * self.cp_world_size :]
        if not external_block_ids:
            return 0, False
        try:
            external_hit_blocks = self.store.lookup_on_prefix(external_block_ids) + 1
            external_hit_blocks //= self.cp_world_size
        except Exception as e:
            external_hit_blocks = 0
            logger.error(
                f"request {request.request_id} look up error. {type(e).__name__}: {e}"
            )

        logger.info_once(
            f"request_id: {request.request_id}, "
            f"total_blocks_num: {len(ucm_block_ids)}, "
            f"hit hbm: {hbm_hit_block_num * self.cp_world_size}, "
            f"hit external: {external_hit_blocks * self.cp_world_size}"
        )
        if self.metrics_config:
            ucmmetrics.update_stats(
                {
                    "interval_lookup_hit_rates": external_hit_blocks
                    * self.cp_world_size
                    / len(ucm_block_ids)
                },
            )

        total_hit_block_num = hbm_hit_block_num + external_hit_blocks

        external_hit_tokens = external_hit_blocks * self.block_size

        # When all the tokens are cached in ssd or hbm,
        # we need to recompute the last token. This if condition will be removed
        # once vLLM scheduler provides a better solution in the future.
        num_total_hit_tokens = total_hit_block_num * self.block_size
        if num_total_hit_tokens == request.num_tokens:
            external_hit_tokens -= 1

        self.requests_meta[request.request_id] = RequestMeta(
            ucm_block_ids=ucm_block_ids,
            hbm_hit_block_num=hbm_hit_block_num,
            total_hit_block_num=total_hit_block_num,
            num_token_ids=len(request.all_token_ids),
            token_processed=num_total_hit_tokens,
        )

        return external_hit_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        pass

    def _generate_dispatch_meta(
        self,
        req_meta: RequestMeta,
        new_tokens: int,
        vllm_block_ids: list[int],
        need_load: bool = True,
    ) -> RequestDispatchMeta:
        """
        Request Blocks layout:
        ----------------------------------------------------------------------------------------------------
        | local_computed_block(HBM hit) | external_computed_block(external hit) | new_block(need to dump)  |
        ----------------------------------------------------------------------------------------------------
        |      hbm_hit_block_num        |                 LOAD                  |     new_blocks_num       |
        ----------------------------------------------------------------------------------------------------
        |                              total_hit_block_num                      |
        ----------------------------------------------------------------------------------------------------
        |                                         scheduled_block_num                                      |
        """

        hbm_hit_block_num = req_meta.hbm_hit_block_num
        total_hit_block_num = req_meta.total_hit_block_num
        ucm_block_ids = req_meta.ucm_block_ids
        req_meta.vllm_block_ids.extend(vllm_block_ids)

        load_ucm_block_ids, load_vllm_block_ids = [], []
        dump_ucm_block_ids, dump_vllm_block_ids = [], []
        if need_load:
            load_ucm_block_ids = ucm_block_ids[
                hbm_hit_block_num
                * self.cp_world_size : total_hit_block_num
                * self.cp_world_size
            ]
            load_vllm_block_ids = vllm_block_ids[hbm_hit_block_num:total_hit_block_num]

        # Track the old token_processed before advancing for mamba dump range.
        old_token_processed = req_meta.token_processed
        will_dump = req_meta.token_processed < req_meta.num_token_ids
        if will_dump:
            start_idx = req_meta.token_processed // self.block_size
            end_idx = (req_meta.token_processed + new_tokens) // self.block_size
            dump_ucm_block_ids = ucm_block_ids[
                start_idx * self.cp_world_size : end_idx * self.cp_world_size
            ]
            dump_vllm_block_ids = req_meta.vllm_block_ids[start_idx:end_idx]
            req_meta.token_processed += new_tokens

        # ---- Mamba per-group load / dump block IDs ----
        # Single-store design: mamba uses the same UCM hash keys as attention
        # (requires mamba_block_size == attention block_size via mamba_cache_mode='align').
        mamba_load: list[tuple[list[bytes], list[int]]] = []
        mamba_dump: list[tuple[list[bytes], list[int]]] = []
        for k, mamba_block_size in enumerate(self.mamba_block_sizes):
            m_vllm = (
                req_meta.mamba_vllm_ids[k]
                if k < len(req_meta.mamba_vllm_ids)
                else []
            )

            # Load: mamba blocks whose corresponding token range was hit externally.
            total_hit_tokens = total_hit_block_num * self.block_size
            hbm_hit_tokens = hbm_hit_block_num * self.block_size
            m_total_hit = total_hit_tokens // mamba_block_size
            m_hbm_hit = hbm_hit_tokens // mamba_block_size
            m_load_ucm: list[bytes] = []
            m_load_vllm: list[int] = []
            if need_load and m_total_hit > m_hbm_hit:
                m_load_ucm = load_ucm_block_ids
                m_load_vllm = m_vllm[m_hbm_hit:m_total_hit]
            mamba_load.append((m_load_ucm, m_load_vllm))

            # Dump: mamba blocks newly covered by the tokens processed this step.
            if will_dump:
                token_start_m = old_token_processed
                token_end_m = old_token_processed + new_tokens
                m_start = token_start_m // mamba_block_size
                m_end = token_end_m // mamba_block_size
                m_dump_ucm = dump_ucm_block_ids
                m_dump_vllm = m_vllm[m_start:m_end]
            else:
                m_dump_ucm, m_dump_vllm = [], []
            mamba_dump.append((m_dump_ucm, m_dump_vllm))

        return RequestDispatchMeta(
            (load_ucm_block_ids, load_vllm_block_ids),
            (dump_ucm_block_ids, dump_vllm_block_ids),
            mamba_load=mamba_load,
            mamba_dump=mamba_dump,
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        requests_dispatch_meta = {}
        # for new request, we need to load and dump
        for request in scheduler_output.scheduled_new_reqs:
            request_id, vllm_block_ids = request.req_id, request.block_ids[0]
            req_meta = self.requests_meta.get(request_id)
            if req_meta:
                # Populate mamba physical block IDs from kv_cache group slots.
                # block_ids[0] = attention group; block_ids[1..] = mamba groups.
                if self.num_mamba_groups and len(request.block_ids) > 1:
                    req_meta.mamba_vllm_ids = [
                        list(request.block_ids[k + 1])
                        for k in range(self.num_mamba_groups)
                        if k + 1 < len(request.block_ids)
                    ]
                requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request_id],
                    vllm_block_ids,
                )

        # for cached request, there are 3 situation:
        # 1. chunked prefill: we only need dump
        # 2. resumed: we need to handle like new request
        # 3. TODO decode stage: nothing happened
        scheduled_cached_reqs = scheduler_output.scheduled_cached_reqs
        if not isinstance(scheduled_cached_reqs, list):
            # >= 0.9.2
            for i, request_id in enumerate(scheduled_cached_reqs.req_ids):
                req_meta = self.requests_meta.get(request_id)
                if req_meta:
                    new_block_ids = []
                    raw_new = scheduled_cached_reqs.new_block_ids[i]
                    if raw_new is not None:
                        new_block_ids = raw_new[0]
                        # Accumulate new mamba block IDs into req_meta.
                        if self.num_mamba_groups and len(raw_new) > 1:
                            for k in range(self.num_mamba_groups):
                                if k + 1 < len(raw_new) and raw_new[k + 1]:
                                    if k >= len(req_meta.mamba_vllm_ids):
                                        req_meta.mamba_vllm_ids.append(list(raw_new[k + 1]))
                                    else:
                                        req_meta.mamba_vllm_ids[k].extend(raw_new[k + 1])
                    if hasattr(scheduled_cached_reqs, "resumed_from_preemption"):
                        resumed_from_preemption = (
                            scheduled_cached_reqs.resumed_from_preemption[i]
                        )
                    else:
                        resumed_from_preemption = (
                            request_id in scheduled_cached_reqs.resumed_req_ids
                        )
                    requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                        req_meta,
                        scheduler_output.num_scheduled_tokens[request_id],
                        new_block_ids,
                        resumed_from_preemption,
                    )
        else:
            for request in scheduled_cached_reqs:
                request_id = request.req_id
                req_meta = self.requests_meta.get(request_id)
                if req_meta:
                    raw_new = request.new_block_ids
                    attn_new = raw_new[0] if raw_new else []
                    if self.num_mamba_groups and raw_new and len(raw_new) > 1:
                        for k in range(self.num_mamba_groups):
                            if k + 1 < len(raw_new) and raw_new[k + 1]:
                                if k >= len(req_meta.mamba_vllm_ids):
                                    req_meta.mamba_vllm_ids.append(list(raw_new[k + 1]))
                                else:
                                    req_meta.mamba_vllm_ids[k].extend(raw_new[k + 1])
                    requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                        req_meta,
                        scheduler_output.num_scheduled_tokens[request_id],
                        attn_new,
                        request.resumed_from_preemption,
                    )

        # clear finished request
        for request_id in scheduler_output.finished_req_ids:
            self.requests_meta.pop(request_id, None)

        return UCMConnectorMetadata(requests_dispatch_meta)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        request_to_task: dict[str, Task] = {}
        is_load = False
        num_loaded_block = 0
        num_loaded_request = 0
        load_start_time = time.perf_counter() * 1000
        for request_id, request in metadata.request_meta.items():
            if len(request.load_block_ids[0]) == 0:
                continue
            is_load = True
            num_loaded_block += len(request.load_block_ids[0])
            num_loaded_request += 1

            ucm_block_ids, vllm_block_ids = request.load_block_ids
            if self.tp_rank != 0 and not self.is_mla:
                for i, ucm_block_id in enumerate(ucm_block_ids):
                    ucm_block_ids[i] = self.request_hasher(ucm_block_id)
            try:
                mamba_vllm_block_ids = self._mamba_vllm_ids_from_dispatch(
                    request.mamba_load
                )
                total_ptrs = self.kv_cache_layout.extract_block_addrs(
                    vllm_block_ids, mamba_vllm_block_ids
                )
                total_ptrs = total_ptrs.reshape(total_ptrs.shape[0], -1)
                shard_indexs = [0] * len(ucm_block_ids)
                task = self.store.load_data(ucm_block_ids, shard_indexs, total_ptrs)
                request_to_task[request_id] = task
            except Exception as e:
                logger.error(
                    f"request {request_id} submit load task error. {type(e).__name__}: {e}"
                )
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[1]
                )
                num_loaded_block -= len(request.load_block_ids[0])

        for request_id, task in request_to_task.items():
            try:
                self.store.wait(task)
            except Exception as e:
                logger.error(
                    f"request {request_id} wait load task error. {type(e).__name__}: {e}"
                )
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[1]
                )
                num_loaded_block -= len(
                    metadata.request_meta[request_id].load_block_ids[0]
                )

        load_end_time = time.perf_counter() * 1000
        load_speed = (
            num_loaded_block
            * self.block_data_size
            / (load_end_time - load_start_time)
            / 1024
            / 1024
        )  # GB/s
        if self.metrics_config and is_load:
            ucmmetrics.update_stats(
                {
                    "load_requests_num": num_loaded_request,
                    "load_blocks_num": num_loaded_block,
                    "load_duration": load_end_time - load_start_time,
                    "load_speed": load_speed,
                }
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def _get_dump_event_handle(self) -> int:
        if not self.enable_event_sync:
            self.device.synchronize()
            return 0

        event_handle = self.device.get_event_handle()
        if event_handle == 0:
            self.device.synchronize()
        return event_handle

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        pass

    def wait_for_save(self) -> None:
        # TODO support PP
        if self.is_mla and self.tp_rank != 0:
            return

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        dump_tasks: List[Task] = []
        is_save = False
        num_saved_block = 0
        num_saved_request = 0
        total_ucm_block_ids, total_vllm_block_ids = [], []
        total_mamba_vllm_block_ids: list[list[int]] = [
            [] for _ in self.mamba_kv_cache_layouts
        ]
        for request_id, request in metadata.request_meta.items():
            if len(request.dump_block_ids[0]) == 0:
                continue
            is_save = True
            num_saved_block += len(request.dump_block_ids[0])
            num_saved_request += 1

            ucm_block_ids, vllm_block_ids = request.dump_block_ids
            if self.tp_rank != 0:
                for i, ucm_block_id in enumerate(ucm_block_ids):
                    ucm_block_ids[i] = self.request_hasher(ucm_block_id)
            total_ucm_block_ids.extend(ucm_block_ids)
            total_vllm_block_ids.extend(vllm_block_ids)
            for k, mamba_vllm_ids in enumerate(
                self._mamba_vllm_ids_from_dispatch(request.mamba_dump)
            ):
                total_mamba_vllm_block_ids[k].extend(mamba_vllm_ids)

        if is_save:
            try:
                total_ptrs = self.kv_cache_layout.extract_block_addrs(
                    total_vllm_block_ids, total_mamba_vllm_block_ids
                )
                total_ptrs = total_ptrs.reshape(total_ptrs.shape[0], -1)
                shard_indexs = [0] * len(total_ucm_block_ids)
                event_handle = self._get_dump_event_handle()
                save_start_time = time.perf_counter() * 1000
                task = self.store.dump_data(
                    total_ucm_block_ids, shard_indexs, total_ptrs, event_handle
                )
                dump_tasks.append(task)
            except Exception as e:
                logger.error(f"dump kv cache failed. {type(e).__name__}: {e}")
                return

            try:
                for task in dump_tasks:
                    self.store.wait(task)
                save_end_time = time.perf_counter() * 1000
            except Exception as e:
                logger.error(f"wait for dump kv cache failed. {type(e).__name__}: {e}")
                return

            save_speed = (
                num_saved_block
                * self.block_data_size
                / (save_end_time - save_start_time)
                / 1024
                / 1024
            )  # GB/s
            if self.metrics_config:
                ucmmetrics.update_stats(
                    {
                        "save_requests_num": num_saved_request,
                        "save_blocks_num": num_saved_block,
                        "save_duration": save_end_time - save_start_time,
                        "save_speed": save_speed,
                    },
                )

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.
        """
        res = self._invalid_block_ids
        self._invalid_block_ids = set()
        return res
    
    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None


class UCMLayerWiseConnector(UCMDirectConnector):
    """
    This Connector means overlap:
    load l0 -> forward l0 -> save l0
               load l1    -> forward l1 -> save l1
                             load l2    -> forward l2 -> save l2
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole,
                 kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        # {layer_id: {request_id: Task}}
        self.load_tasks: dict[int, dict[str, Task]] = defaultdict(dict)
        self.dump_tasks: dict[str, Task] = {}
        self.use_layerwise = True
        self.is_save = False
        self.need_load = False
        self.dump_total_ptrs: np.ndarray | None = None
        self.request_data: list[tuple[str, list, np.ndarray]] = []
        self._failure_req_ids: set[str] = set()
        logger.info("Init UCMLayerWiseConnector.")

    def _submit_request_load_tasks_for_layer(
        self,
        layer_id: int,
        local_row: int,
        metadata: "UCMConnectorMetadata",
    ) -> None:
        for request_id, ucm_block_ids, total_ptrs in self.request_data:
            if request_id in self._failure_req_ids:
                continue
            try:
                shard_indexs = [layer_id] * len(ucm_block_ids)
                layer_ptrs = total_ptrs[local_row]
                task = self.store.load_data(ucm_block_ids, shard_indexs, layer_ptrs)
                self.load_tasks[layer_id][request_id] = task
            except Exception as e:
                logger.error(
                    f"request {request_id} submit load task error. {type(e).__name__}: {e}"
                )
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[1]
                )
                self._failure_req_ids.add(request_id)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        self.load_tasks.clear()
        self.request_data.clear()
        self._failure_req_ids.clear()
        self.need_load = False

        for request_id, request in metadata.request_meta.items():
            if len(request.load_block_ids[0]) == 0:
                continue

            self.need_load = True
            ucm_block_ids, vllm_block_ids = request.load_block_ids
            if self.tp_rank % self.tp_size != 0 and not self.is_mla:
                for i, ucm_block_id in enumerate(ucm_block_ids):
                    ucm_block_ids[i] = self.request_hasher(ucm_block_id)
            total_ptrs = self.kv_cache_layout.extract_block_addrs(
                vllm_block_ids, layer_first=True
            )
            self.request_data.append((request_id, ucm_block_ids, total_ptrs))

        if self.need_load:
            self._submit_request_load_tasks_for_layer(
                self.first_layer_id,
                self.layer_id_to_row[self.first_layer_id],
                metadata,
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._connector_metadata:
            return
        if not self.need_load:
            return
        metadata = self._get_connector_metadata()
        current_layer_id = self.layer_name_to_id[layer_name]

        # Pop before wait so MTP / rollback paths that revisit the same layer_name
        # do not call store.wait() again on already-completed handles.
        layer_tasks = self.load_tasks.pop(current_layer_id, {})
        for request_id, task in layer_tasks.items():
            try:
                self.store.wait(task)
            except Exception as e:
                logger.error(
                    f"request {request_id} wait {layer_name} load failed. {type(e).__name__}: {e}"
                )
                self._invalid_block_ids.update(
                    metadata.request_meta[request_id].load_block_ids[1]
                )
                self._failure_req_ids.add(request_id)

        next_layer_id = current_layer_id + 1
        if next_layer_id not in self.layer_ids:
            return
        next_local_row = self.layer_id_to_row[next_layer_id]

        self._submit_request_load_tasks_for_layer(
            next_layer_id, next_local_row, metadata
        )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        if not self._connector_metadata:
            return
        if self.is_mla and self.tp_rank % self.tp_size != 0:
            return

        metadata = self._get_connector_metadata()

        total_ucm_block_ids, total_vllm_block_ids = [], []
        layer_id = self.layer_name_to_id[layer_name]
        local_layer_id = self.layer_id_to_row[layer_id]
        for _, request in metadata.request_meta.items():
            if len(request.dump_block_ids[0]) == 0:
                continue

            self.is_save = True
            ucm_block_ids, vllm_block_ids = request.dump_block_ids
            if self.tp_rank % self.tp_size != 0 and local_layer_id == 0:
                for i, ucm_block_id in enumerate(ucm_block_ids):
                    ucm_block_ids[i] = self.request_hasher(ucm_block_id)
            total_ucm_block_ids.extend(ucm_block_ids)
            total_vllm_block_ids.extend(vllm_block_ids)

        if self.is_save:
            if self.dump_total_ptrs is None:
                self.dump_total_ptrs = self.kv_cache_layout.extract_block_addrs(
                    total_vllm_block_ids, layer_first=True
                )
            shard_indexs = [layer_id] * len(total_ucm_block_ids)
            try:
                layer_ptrs = np.ascontiguousarray(self.dump_total_ptrs[local_layer_id])
                event_handle = self._get_dump_event_handle()
                task = self.store.dump_data(
                    total_ucm_block_ids, shard_indexs, layer_ptrs, event_handle
                )
                self.dump_tasks[layer_name] = task
            except Exception as e:
                logger.error(f"submit dump task failed. {type(e).__name__}: {e}")

    def wait_for_save(self) -> None:
        if not self.is_save:
            return
        try:
            for layer_name in self.kv_caches:
                if layer_name in self.dump_tasks:
                    self.store.wait(self.dump_tasks[layer_name])
        except Exception as e:
            logger.error(f"wait for dump kv cache failed. {type(e).__name__}: {e}")
        self.dump_tasks.clear()
        self.is_save = False
        self.dump_total_ptrs = None
        if self.enable_event_sync:
            self.device.destroy_event_handles()


class UCMCPConnector(UCMLayerWiseConnector):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole,
                 kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        self.use_layerwise = self.launch_config.get("use_layerwise", False)

        try:
            from vllm.distributed import get_dcp_group, get_pcp_group
        except ImportError as e:
            raise ImportError(
                "Please check if the current vLLM version supports DCP and PCP features."
            ) from e

        try:
            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = (
                get_pcp_group().rank_in_group if self.pcp_world_size > 1 else 0
            )
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
            self.pcp_world_size = 1
            self.pcp_rank = 0
        self.cp_world_size = (
            self._vllm_config.parallel_config.prefill_context_parallel_size
            * self._vllm_config.parallel_config.decode_context_parallel_size
        )
        self.current_rank = self.dcp_world_size * self.pcp_rank + self.dcp_rank
        old_tp_size = vllm_config.parallel_config.tensor_parallel_size
        logger.info(
            f"pcp_world_size: {self.pcp_world_size}, pcp_rank: {self.pcp_rank}, dcp_world_size: {self.dcp_world_size}, dcp_rank: {self.dcp_rank}"
        )

        self.tp_rank %= self.tp_size
        self.tp_rank //= self.dcp_world_size
        if not self.is_mla:
            vllm_config.parallel_config.tensor_parallel_size //= self.dcp_world_size

        if role == KVConnectorRole.SCHEDULER:
            self.request_hasher = RequestHasher(vllm_config, 0)
            self._seed = self.request_hasher("UCM_HASH_SEED")
            # init scheduler-size connector
            self.store = self._create_store(None)
        else:
            self.request_hasher = RequestHasher(vllm_config, self.tp_rank)
        vllm_config.parallel_config.tensor_parallel_size = old_tp_size
        self.block_size *= self.cp_world_size
        logger.info("Init UCMCPConnector.")

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        # When DCP/PCP features are enabled,
        # the blocks that each device can process are [current_rank :: cp_world_size],
        # where current_rank = self.dcp_world_size * self.pcp_rank + self.dcp_rank.
        for _, request in connector_metadata.request_meta.items():
            if len(request.load_block_ids[0]) > 0:
                ucm_block_ids, vllm_block_ids = request.load_block_ids
                ucm_block_ids = ucm_block_ids[self.current_rank :: self.cp_world_size]
                request.load_block_ids = (ucm_block_ids, vllm_block_ids)

            if len(request.dump_block_ids[0]) > 0:
                ucm_block_ids, vllm_block_ids = request.dump_block_ids
                ucm_block_ids = ucm_block_ids[self.current_rank :: self.cp_world_size]
                request.dump_block_ids = (ucm_block_ids, vllm_block_ids)
        super().bind_connector_metadata(connector_metadata)

    def start_load_kv(self, forward_context, **kwargs):
        if self.use_layerwise:
            super().start_load_kv(forward_context, **kwargs)
        else:
            super(UCMLayerWiseConnector, self).start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if self.use_layerwise:
            super().wait_for_layer_load(layer_name)
        else:
            pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        if self.use_layerwise:
            super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)
        else:
            pass

    def wait_for_save(self):
        if self.use_layerwise:
            super().wait_for_save()
        else:
            super(UCMLayerWiseConnector, self).wait_for_save()


class UCMPDConnector(UCMDirectConnector):
    """
    This Connector means overlap (especially for Decode Instance):
    step (req0,1,2) forward -> step (req0,1,2,3) forward
    load req3               -> load req4
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config, role)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        raise NotImplementedError

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        raise NotImplementedError


class UCMMockConnector(UCMDirectConnector):
    """
    This Connector can control hit ratio, for example: if your hit ratio is 100%,
    you can set "hit_ratio" by config or env_vars, then get_num_new_matched_tokens()
    will reduce hit_tokens under the hit_ratio you set.
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config, role)
        self._hit_ratio = float(self.launch_config["hit_ratio"])
        logger.info(f"hit_ratio: {self._hit_ratio}")

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        hit_tokens, _ = super().get_num_new_matched_tokens(request, num_computed_tokens)
        expect_hit_tokens = int(self._hit_ratio * request.num_prompt_tokens)
        if hit_tokens <= expect_hit_tokens:
            return hit_tokens, False
        expect_hit_block_num = expect_hit_tokens // self.block_size
        request_meta = self.requests_meta[request.request_id]
        request_meta.total_hit_block_num = expect_hit_block_num
        request_meta.hbm_hit_block_num = min(
            expect_hit_block_num, request_meta.hbm_hit_block_num
        )

        logger.info(
            "Hijacked By MockConnector,"
            f"request_id: {request.request_id}, "
            f"total_blocks_num: {len(request_meta.ucm_block_ids)}, "
            f"hit hbm: {request_meta.hbm_hit_block_num}, "
            f"hit external: {request_meta.total_hit_block_num - request_meta.hbm_hit_block_num}"
        )

        return expect_hit_block_num * self.block_size, False


class UCMLiteConnector(UCMDirectConnector):
    def __init__(self, vllm_config, role):
        ucm_config = Config(vllm_config.kv_transfer_config)
        launch_config = ucm_config.get_config()
        enable_record_traces = launch_config.get("enable_record_traces", False)
        persist_token_threshold = launch_config.get("persist_token_threshold", 0)
        vllm_config.kv_transfer_config.kv_connector_extra_config = {
            "ucm_connectors": [
                {
                    "ucm_connector_name": "UcmPipelineStore",
                    "ucm_connector_config": {
                        "store_pipeline": "Fake",
                        "share_buffer_enable": True,
                        "buffer_number": 244032232,
                    },
                }
            ],
            "enable_record_traces": enable_record_traces,
            "persist_token_threshold": persist_token_threshold,
            "use_lite": True,
        }
        super().__init__(vllm_config, role)
        self.total_block_nums = 0
        self.total_hit_block_nums = 0
        logger.info("Init UCMLiteConnector.")

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        super().get_num_new_matched_tokens(request, num_computed_tokens)

        external_hit_blocks = 0
        req_blocks_num = len(request.all_token_ids) // self.hash_block_size
        if req_blocks_num < 1:
            return 0, False
        self.total_block_nums += req_blocks_num
        if request.request_id in self.requests_meta:
            request_meta = self.requests_meta[request.request_id]
            external_hit_blocks = (
                request_meta.total_hit_block_num - request_meta.hbm_hit_block_num
            )
            need_dump_blks = request_meta.ucm_block_ids[
                request_meta.total_hit_block_num :
            ]
            shard_indexs = [0] * len(need_dump_blks)
            total_ptrs = [[0]] * len(need_dump_blks)
            try:
                task = self.store.dump_data(need_dump_blks, shard_indexs, total_ptrs)
                self.store.wait(task)
            except Exception as e:
                logger.error(
                    f"request {request.request_id} wait dump task error. {type(e).__name__}: {e}"
                )
            self.requests_meta[request.request_id] = RequestMeta()

        self.total_hit_block_nums += external_hit_blocks

        logger.info(
            f"req external hit rate: {(external_hit_blocks / req_blocks_num):.2f}, "
            f"total external hit rate: {(self.total_hit_block_nums / self.total_block_nums):.2f}"
        )
        return 0, False


class UCMConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole,
                 kv_cache_config=None):
        super().__init__(vllm_config=vllm_config, role=role)
        self.connector: KVConnectorBase_V1
        ucm_config = Config(vllm_config.kv_transfer_config)
        self.launch_config = ucm_config.get_config()
        logger.info(f"self.launch_config: {self.launch_config}")

        use_layerwise = (
            self.launch_config.get("use_layerwise", False)
            if self.launch_config is not None
            else False
        )

        pp_enabled = self._vllm_config.parallel_config.pipeline_parallel_size > 1
        if pp_enabled and not use_layerwise:
            raise RuntimeError(
                "Pipeline parallelism is not supported in UCMDirectConnector, please set use_layerwise=True."
            )

        use_lite = (
            self.launch_config.get("use_lite", False)
            if self.launch_config is not None
            else False
        )

        use_ratio_rate = (
            self.launch_config is not None and "hit_ratio" in self.launch_config
        )

        use_cp_parallel = (
            hasattr(self._vllm_config.parallel_config, "prefill_context_parallel_size")
            and hasattr(
                self._vllm_config.parallel_config, "decode_context_parallel_size"
            )
            and self._vllm_config.parallel_config.prefill_context_parallel_size
            * self._vllm_config.parallel_config.decode_context_parallel_size
            > 1
        )

        if use_lite:
            self.connector = UCMLiteConnector(vllm_config, role)
        elif use_ratio_rate:
            self.connector = UCMMockConnector(vllm_config, role)
        elif use_cp_parallel:
            self.connector = UCMCPConnector(vllm_config, role, kv_cache_config)
        elif use_layerwise:
            self.connector = UCMLayerWiseConnector(vllm_config, role, kv_cache_config)
        else:
            self.connector = UCMDirectConnector(vllm_config, role, kv_cache_config)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        return self.connector.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        self.connector.update_state_after_alloc(request, blocks, num_external_tokens)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args: kv_caches:
            dictionary of layer names, kv cache
        """
        self.connector.register_kv_caches(kv_caches)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        return self.connector.build_connector_meta(scheduler_output)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.

        Args:
            connector_metadata (dict): the connector metadata.
        """
        self.connector.bind_connector_metadata(connector_metadata)

    def has_connector_metadata(self) -> bool:
        """Check whether the connector metadata is currently set.

        Returns:
            bool: True if connector metadata exists, False otherwise.
        """
        return self.connector.has_connector_metadata()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        self.connector.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        self.connector.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        self.connector.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        self.connector.wait_for_save()

    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata.

        This function should be called by the model runner every time
        after the model execution.
        """
        self.connector.clear_connector_metadata()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.
        """
        return self.connector.get_block_ids_with_load_errors()

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.connector.request_finished_all_groups(request, block_ids)
    