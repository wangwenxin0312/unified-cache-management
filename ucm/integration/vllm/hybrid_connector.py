import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, MambaSpec

from ucm.integration.vllm.device import create_device
from ucm.integration.vllm.ucm_connector import (
    RequestDispatchMeta,
    RequestMeta,
    UCMConnectorMetadata,
    UCMDirectConnector,
)
from ucm.logger import init_logger
from ucm.sparse.state import has_ucm_sparse

if TYPE_CHECKING:
    from vllm.v1.request import Request


logger = init_logger(__name__)


@dataclass
class _HybridRow:
    conv_or_padding_ptr: int
    k_or_ssm_ptr: int
    v_or_padding_ptr: int
    conv_or_padding_size: int
    k_or_ssm_size: int
    v_or_padding_size: int


class HybridKVCacheLayout:
    """Physical Ascend hybrid attention + mamba KV layout.

    A single physical block stores one cache group in this segment order:
    [conv_or_padding, k_or_ssm, v_or_padding]. Attention and mamba groups
    share the same raw tensor row, while their block ids select different
    physical blocks inside that row.
    """

    def __init__(
        self,
        kvcaches: dict[str, Any],
        vllm_config: "VllmConfig",
        kv_cache_config: Optional["KVCacheConfig"],
    ) -> None:
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.layer_name_to_id = {
            name: extract_layer_index(name) for name in kvcaches.keys()
        }
        self.first_layer_id = min(self.layer_name_to_id.values())
        self.base_ptrs: np.ndarray
        self.tensor_size_lists: np.ndarray
        self._build_layout(kvcaches)

    @staticmethod
    def _is_tensor_sequence(value: Any) -> bool:
        return isinstance(value, (tuple, list)) and all(
            isinstance(tensor, torch.Tensor) for tensor in value
        )

    @staticmethod
    def _block_stride_bytes(tensor: torch.Tensor) -> int:
        if tensor.dim() < 2:
            raise ValueError(f"Unsupported hybrid cache tensor shape: {tensor.shape}")
        return int(tensor.stride(0) * tensor.element_size())

    def _layer_specs(self) -> dict[str, Any]:
        if self.kv_cache_config is None:
            return {}
        specs: dict[str, Any] = {}
        for group in self.kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                specs[layer_name] = group.kv_cache_spec
        return specs

    def _iter_shared_rows(self, kvcaches: dict[str, Any]) -> Iterable[list[str]]:
        if self.kv_cache_config is not None:
            for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
                row_names = [
                    layer_name
                    for layer_name in kv_cache_tensor.shared_by
                    if layer_name in kvcaches
                ]
                if row_names:
                    yield row_names
            return

        # Fallback for older vLLM signatures: common hybrid models repeat
        # mamba/linear layers followed by one attention layer.
        by_layer_id = {
            extract_layer_index(layer_name): layer_name for layer_name in kvcaches
        }
        for layer_id in sorted(by_layer_id):
            attn_name = by_layer_id[layer_id]
            if "self_attn" not in attn_name:
                continue
            row_names = [
                by_layer_id[i]
                for i in range(layer_id - 3, layer_id + 1)
                if i in by_layer_id
            ]
            if row_names:
                yield row_names

    def _classify_row(
        self,
        row_names: Sequence[str],
        kvcaches: dict[str, Any],
        layer_specs: dict[str, Any],
    ) -> tuple[str, str]:
        attn_name = ""
        mamba_name = ""
        for layer_name in row_names:
            value = kvcaches[layer_name]
            spec = layer_specs.get(layer_name)
            if isinstance(spec, AttentionSpec) or "self_attn" in layer_name:
                attn_name = layer_name
            elif isinstance(spec, MambaSpec) or "linear_attn" in layer_name:
                mamba_name = mamba_name or layer_name
            elif self._is_tensor_sequence(value):
                if len(value) >= 2 and value[0].dim() >= 4:
                    attn_name = layer_name
                elif len(value) >= 2:
                    mamba_name = mamba_name or layer_name

        if not attn_name or not mamba_name:
            raise ValueError(
                "Hybrid KV cache row must contain both attention and mamba "
                f"layers, got row_names={list(row_names)}"
            )
        return attn_name, mamba_name

    def _build_row(
        self,
        attn_cache: Any,
        mamba_cache: Any,
    ) -> _HybridRow:
        if not self._is_tensor_sequence(attn_cache) or len(attn_cache) < 2:
            raise TypeError(f"Unsupported attention cache type: {type(attn_cache)}")
        if not self._is_tensor_sequence(mamba_cache) or len(mamba_cache) < 2:
            raise TypeError(f"Unsupported mamba cache type: {type(mamba_cache)}")

        conv_tensor = mamba_cache[0]
        ssm_tensor = mamba_cache[1]
        k_tensor = attn_cache[0]
        v_tensor = attn_cache[1]

        conv_size = self._block_stride_bytes(conv_tensor)
        ssm_size = self._block_stride_bytes(ssm_tensor)
        k_size = self._block_stride_bytes(k_tensor)
        v_size = self._block_stride_bytes(v_tensor)
        middle_size = max(ssm_size, k_size, v_size)

        if k_tensor.data_ptr() != ssm_tensor.data_ptr():
            logger.warning(
                "Hybrid middle segment pointers differ: "
                f"k={k_tensor.data_ptr()}, ssm={ssm_tensor.data_ptr()}. "
                "Using the mamba ssm pointer as the physical segment base."
            )

        return _HybridRow(
            conv_or_padding_ptr=conv_tensor.data_ptr(),
            k_or_ssm_ptr=ssm_tensor.data_ptr(),
            v_or_padding_ptr=v_tensor.data_ptr(),
            conv_or_padding_size=conv_size,
            k_or_ssm_size=middle_size,
            v_or_padding_size=middle_size,
        )

    def _build_layout(self, kvcaches: dict[str, Any]) -> None:
        layer_specs = self._layer_specs()
        rows: list[_HybridRow] = []
        for row_names in self._iter_shared_rows(kvcaches):
            attn_name, mamba_name = self._classify_row(
                row_names, kvcaches, layer_specs
            )
            rows.append(self._build_row(kvcaches[attn_name], kvcaches[mamba_name]))

        if not rows:
            raise ValueError("No hybrid attention + mamba KV cache rows were detected.")

        self.base_ptrs = np.asarray(
            [
                [
                    row.conv_or_padding_ptr,
                    row.k_or_ssm_ptr,
                    row.v_or_padding_ptr,
                ]
                for row in rows
            ],
            dtype=np.uint64,
        )
        self.tensor_size_lists = np.asarray(
            [
                [
                    row.conv_or_padding_size,
                    row.k_or_ssm_size,
                    row.v_or_padding_size,
                ]
                for row in rows
            ],
            dtype=np.uint64,
        )

        logger.info(
            "Hybrid attention + mamba base_ptrs: %s, tensor_size_lists: %s, block_size: %d",
            self.base_ptrs.shape,
            self.tensor_size_lists.shape,
            self.block_size,
        )

    def extract_block_addrs(
        self, vllm_block_ids: list[int], layer_first: bool = False
    ) -> np.ndarray:
        vllm_block_ids_np = np.asarray(vllm_block_ids, dtype=np.uint64)
        if layer_first:
            return (
                self.tensor_size_lists[:, None, :] * vllm_block_ids_np[None, :, None]
                + self.base_ptrs[:, None, :]
            )
        return (
            vllm_block_ids_np[:, None, None] * self.tensor_size_lists[None, :, :]
            + self.base_ptrs[None, :, :]
        )

    @property
    def tensor_size_list(self) -> list[int]:
        return self.tensor_size_lists.reshape(-1).tolist()

    @property
    def shard_size(self) -> int:
        return int(self.tensor_size_lists.sum())

    @property
    def block_size(self) -> int:
        return int(self.tensor_size_lists.sum())


class UCMHybridConnector(UCMDirectConnector, SupportsHMA):
    """UCM connector for Ascend hybrid attention + mamba layout."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        self.kv_cache_config = kv_cache_config
        self.attention_group_indices, self.mamba_group_indices = (
            self._build_group_indices(kv_cache_config)
        )
        self.num_kv_cache_groups = self._get_num_kv_cache_groups(kv_cache_config)
        super().__init__(vllm_config, role)
        if self.cp_world_size != 1:
            raise RuntimeError("UCMHybridConnector does not support CP yet.")

    @staticmethod
    def _build_group_indices(
        kv_cache_config: Optional["KVCacheConfig"],
    ) -> tuple[list[int], list[int]]:
        if kv_cache_config is None:
            return [0], [1]

        attention_group_indices = []
        mamba_group_indices = []
        for group_index, group in enumerate(kv_cache_config.kv_cache_groups):
            if isinstance(group.kv_cache_spec, AttentionSpec):
                attention_group_indices.append(group_index)
            elif isinstance(group.kv_cache_spec, MambaSpec):
                mamba_group_indices.append(group_index)

        if not attention_group_indices or not mamba_group_indices:
            raise ValueError(
                "Hybrid connector requires both attention and mamba KV cache groups."
            )
        return attention_group_indices, mamba_group_indices

    @staticmethod
    def _get_num_kv_cache_groups(kv_cache_config: Optional["KVCacheConfig"]) -> int:
        if kv_cache_config is None:
            return 2
        return len(kv_cache_config.kv_cache_groups)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if has_ucm_sparse() and os.getenv("VLLM_HASH_ATTENTION") == "1":
            self.kv_caches = {
                layer_name: value[0] for layer_name, value in kv_caches.items()
            }
        else:
            self.kv_caches = kv_caches

        sample_kv_layer = next(iter(self.kv_caches.values()))
        if self.kv_cache_dtype is None:
            if isinstance(sample_kv_layer, torch.Tensor):
                self.kv_cache_dtype = sample_kv_layer.dtype
            else:
                self.kv_cache_dtype = sample_kv_layer[0].dtype

        if isinstance(sample_kv_layer, torch.Tensor):
            logger.info("kv cache shape %s", sample_kv_layer.shape)
        elif isinstance(sample_kv_layer, (tuple, list)):
            for i, tensor in enumerate(sample_kv_layer):
                logger.info("kv cache shape %d: %s", i, tensor.shape)
        else:
            raise TypeError(f"Unsupported kv cache type: {type(sample_kv_layer)}")

        self.kv_cache_layout = HybridKVCacheLayout(
            self.kv_caches, self._vllm_config, self.kv_cache_config
        )
        self.block_data_size = self.kv_cache_layout.block_size
        self.layer_name_to_id = self.kv_cache_layout.layer_name_to_id
        self.layer_ids = sorted(set(self.layer_name_to_id.values()))
        self.first_layer_id = self.layer_ids[0]

        self.device = create_device()
        enable_affinity = os.getenv("VLLM_CPU_AFFINITY") == "1"
        worker_cores, store_cores = (
            self.device.split_cores(self.local_rank)
            if enable_affinity
            else (None, None)
        )

        self.store = self._create_store(self.kv_cache_layout, store_cores)

        if worker_cores:
            try:
                os.sched_setaffinity(0, worker_cores)
                logger.info("[VLLM CPU Affinity] Worker bound to cores %s", worker_cores)
            except Exception as e:
                logger.warning("Failed to bind worker: %s", e)

        if self.device is None:
            raise RuntimeError("Unsupported device platform for UCMHybridConnector.")

    def _mamba_ucm_block_id(
        self, attention_ucm_block_id: bytes, group_index: int
    ) -> bytes:
        return self.request_hasher(
            (b"hybrid_mamba_group", group_index, attention_ucm_block_id)
        )

    def _attention_ucm_block_id(
        self, attention_ucm_block_id: bytes, group_index: int
    ) -> bytes:
        if group_index == self.attention_group_indices[0]:
            return attention_ucm_block_id
        return self.request_hasher(
            (b"hybrid_attention_group", group_index, attention_ucm_block_id)
        )

    @staticmethod
    def _normalize_group_block_ids(
        group_block_ids: tuple[list[int], ...] | list[list[int]] | None,
    ) -> tuple[list[int], ...]:
        if group_block_ids is None:
            return tuple()
        return tuple(list(group) for group in group_block_ids)

    def _get_req_group_block_ids(self, req_meta: RequestMeta) -> list[list[int]]:
        group_block_ids = getattr(req_meta, "hybrid_group_block_ids", None)
        if group_block_ids is None:
            group_block_ids = []
            setattr(req_meta, "hybrid_group_block_ids", group_block_ids)
        return group_block_ids

    def _update_req_group_block_ids(
        self,
        req_meta: RequestMeta,
        group_block_ids: tuple[list[int], ...],
        overwrite: bool,
    ) -> None:
        if len(group_block_ids) != self.num_kv_cache_groups:
            raise ValueError(
                "Hybrid connector group count mismatch: expected "
                f"{self.num_kv_cache_groups}, got {len(group_block_ids)}"
            )

        stored_group_block_ids = self._get_req_group_block_ids(req_meta)
        if overwrite or not stored_group_block_ids:
            stored_group_block_ids[:] = [list(group) for group in group_block_ids]
            return

        if len(stored_group_block_ids) != len(group_block_ids):
            raise ValueError(
                "Hybrid request group count changed from "
                f"{len(stored_group_block_ids)} to {len(group_block_ids)}"
            )
        for stored, new in zip(stored_group_block_ids, group_block_ids):
            stored.extend(new)

    def _select_physical_blocks(
        self,
        req_meta: RequestMeta,
        start_idx: int,
        end_idx: int,
    ) -> tuple[list[bytes], list[int]]:
        if start_idx >= end_idx:
            return [], []

        group_block_ids = self._get_req_group_block_ids(req_meta)
        if len(group_block_ids) != self.num_kv_cache_groups:
            raise ValueError(
                "Hybrid request metadata is missing the expected group block ids."
            )

        physical_ucm_ids = []
        physical_block_ids = []
        for group_index in self.attention_group_indices:
            attention_block_ids = group_block_ids[group_index][start_idx:end_idx]
            for block_offset, block_id in enumerate(attention_block_ids):
                token_block_index = start_idx + block_offset
                if token_block_index >= len(req_meta.ucm_block_ids):
                    continue
                physical_ucm_ids.append(
                    self._attention_ucm_block_id(
                        req_meta.ucm_block_ids[token_block_index], group_index
                    )
                )
                physical_block_ids.append(block_id)

        # Keep this in sync with vLLM's mamba_get_block_table_tensor() for
        # mamba_cache_mode="align": start_index = (seq_lens - 1) // block_size.
        # Here end_idx is the number of full logical token blocks represented by
        # the transfer range, so seq_lens=end_idx * block_size and the selected
        # mamba block-table column is end_idx - 1.
        mamba_block_table_index = end_idx - 1
        if mamba_block_table_index < start_idx:
            return physical_ucm_ids, physical_block_ids

        for group_index in self.mamba_group_indices:
            mamba_block_ids = group_block_ids[group_index]
            if mamba_block_table_index >= len(mamba_block_ids):
                continue
            block_id = mamba_block_ids[mamba_block_table_index]
            if block_id == 0:
                continue
            if mamba_block_table_index >= len(req_meta.ucm_block_ids):
                continue
            physical_ucm_ids.append(
                self._mamba_ucm_block_id(
                    req_meta.ucm_block_ids[mamba_block_table_index], group_index
                )
            )
            physical_block_ids.append(block_id)

        return physical_ucm_ids, physical_block_ids

    def _generate_dispatch_meta(
        self,
        req_meta: RequestMeta,
        new_tokens: int,
        group_block_ids: tuple[list[int], ...],
        need_load: bool = True,
    ) -> RequestDispatchMeta:
        self._update_req_group_block_ids(req_meta, group_block_ids, need_load)

        hbm_hit_block_num = req_meta.hbm_hit_block_num
        total_hit_block_num = req_meta.total_hit_block_num

        load_ucm_block_ids, load_vllm_block_ids = [], []
        if need_load:
            load_ucm_block_ids, load_vllm_block_ids = self._select_physical_blocks(
                req_meta, hbm_hit_block_num, total_hit_block_num
            )

        dump_ucm_block_ids, dump_vllm_block_ids = [], []
        if req_meta.token_processed < req_meta.num_token_ids:
            start_idx = req_meta.token_processed // self.block_size
            end_idx = (req_meta.token_processed + new_tokens) // self.block_size
            dump_ucm_block_ids, dump_vllm_block_ids = self._select_physical_blocks(
                req_meta, start_idx, end_idx
            )
            req_meta.token_processed += new_tokens

        return RequestDispatchMeta(
            (load_ucm_block_ids, load_vllm_block_ids),
            (dump_ucm_block_ids, dump_vllm_block_ids),
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        requests_dispatch_meta = {}
        for request in scheduler_output.scheduled_new_reqs:
            request_id = request.req_id
            req_meta = self.requests_meta.get(request_id)
            if req_meta:
                requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request_id],
                    self._normalize_group_block_ids(request.block_ids),
                    True,
                )

        scheduled_cached_reqs = scheduler_output.scheduled_cached_reqs
        if not isinstance(scheduled_cached_reqs, list):
            for i, request_id in enumerate(scheduled_cached_reqs.req_ids):
                req_meta = self.requests_meta.get(request_id)
                if req_meta:
                    new_block_ids = tuple([] for _ in range(self.num_kv_cache_groups))
                    if scheduled_cached_reqs.new_block_ids[i] is not None:
                        new_block_ids = self._normalize_group_block_ids(
                            scheduled_cached_reqs.new_block_ids[i]
                        )
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
                    requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                        req_meta,
                        scheduler_output.num_scheduled_tokens[request_id],
                        self._normalize_group_block_ids(request.new_block_ids),
                        request.resumed_from_preemption,
                    )

        for request_id in scheduler_output.finished_req_ids:
            self.requests_meta.pop(request_id, None)

        return UCMConnectorMetadata(requests_dispatch_meta)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        self.requests_meta.pop(request.request_id, None)
        return False, None


Qwen3NextHybridKVCacheLayout = HybridKVCacheLayout
Qwen3NextUCMConnector = UCMHybridConnector
UCMQwen3NextConnector = UCMHybridConnector


class UCMConnector(UCMHybridConnector):
    pass
