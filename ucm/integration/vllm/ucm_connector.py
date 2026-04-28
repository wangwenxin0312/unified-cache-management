import copy
import hashlib
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
import torch
import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import get_tp_group, get_world_group
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig

from ucm.integration.vllm.device import create_device
from ucm.integration.vllm.kv_cache_layout import (
    KVCacheLayout,
    KVCacheType,
    create_kv_cache_layout,
)
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

logger = init_logger(__name__)


@dataclass
class RequestMeta:
    ucm_block_ids: list[bytes] = field(default_factory=list)
    # Mamba KV group_id -> UCM block hash for that group (paired with last matched attn block).
    mamba_block_ids: dict[int, bytes] = field(default_factory=dict)
    hbm_hit_block_num: int = 0
    # local_computed_block + external_computed_block
    total_hit_block_num: int = 0
    num_token_ids: int = 0
    vllm_block_ids: list[int] = field(default_factory=list)
    token_processed: int = 0
    vllm_block_ids_by_group: tuple[list[int], ...] = field(default_factory=tuple)


@dataclass
class RequestDispatchMeta:
    load_block_ids: tuple[tuple[list[bytes], list[int]], ...]
    dump_block_ids: tuple[tuple[list[bytes], list[int]], ...]


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

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
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

        self.store: Optional[UcmKVStoreBaseV1] = None

        # save block info, avoid hash request twice, and track them until request finished
        self.requests_meta: dict[str, RequestMeta] = {}

        ucm_config = Config(vllm_config.kv_transfer_config)
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self.launch_config = ucm_config.get_config()
        self.connector_configs = self.launch_config.get("ucm_connectors", [])
        self.enable_event_sync = self.launch_config.get("enable_event_sync", True)
        assert len(self.connector_configs) > 0, "no storage connector name in config."

        self.chunk_size = self.block_size
        self.blocks_per_chunk = self.chunk_size // self.block_size
        self.kv_cache_config = kv_cache_config
        self.kv_cache_layout = create_kv_cache_layout(
            self._vllm_config,
            ucm_config,
            getattr(self, "kv_cache_config", None),
        )

        if role == KVConnectorRole.SCHEDULER:
            self.request_hasher = RequestHasher(vllm_config, 0)
            self._seed = self.request_hasher("UCM_HASH_SEED")
            self.store = self._create_store(self.kv_cache_layout, None)
        else:
            self.request_hasher = RequestHasher(
                vllm_config, self.tp_rank % self.tp_size
            )

        self.metrics_config = self.launch_config.get("metrics_config_path", "")
        if self.metrics_config:
            worker_id = (
                get_world_group().rank
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

        # invalid block ids due to load errors
        self._invalid_block_ids: set[int] = set()

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

    def _generate_mamba_block_ids(self, attn_block_id: bytes) -> dict[int, bytes]:
        if not hasattr(self, "kv_cache_layout"):
            return {}
        mamba_group_ids = self.kv_cache_layout.group_ids_by_kv_cache_type.get(
            KVCacheType.MAMBA, []
        )
        return {
            group_id: self.request_hasher((attn_block_id, f"group_{group_id}"))
            for group_id in mamba_group_ids
        }

    def _validate_attn_hit_with_mamba(
        self, attn_block_id: bytes
    ) -> tuple[bool, dict[int, bytes]]:
        mamba_by_group = self._generate_mamba_block_ids(attn_block_id)

        lookup_result = self.store.lookup(list(mamba_by_group.values()))
        if all(lookup_result):
            return True, mamba_by_group
        return False, {}

    def _create_store(
        self,
        kv_cache_layout: Optional[KVCacheLayout],
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
        if self._role == KVConnectorRole.WORKER and kv_cache_layout is not None:
            config["device_id"] = self.local_rank
            config["tensor_size_list"] = (
                kv_cache_layout.tensor_size_list() * self.blocks_per_chunk
            )
            config["shard_size"] = kv_cache_layout.shard_size() * self.blocks_per_chunk
            config["block_size"] = kv_cache_layout.block_size() * self.blocks_per_chunk
            config["local_rank_size"] = self.tp_size if self.is_mla else 1
            if cpu_affinity_cores:
                config["cpu_affinity_cores"] = list(cpu_affinity_cores)
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
        elif isinstance(sample_kv_layer, Tuple):
            # vllm_ascend >= 0.10.0 uses Tuple for kvcaches
            for i, tensor in enumerate(sample_kv_layer):
                logger.info(f"kv cache shape {i}: {tensor.shape}")
        self.kv_cache_layout.initialize_kv_cache_layout(self.kv_caches)
        self.block_data_size = self.kv_cache_layout.block_size()
        self.layer_name_to_id = self.kv_cache_layout.layer_name_to_id
        self.layer_ids = self.kv_cache_layout.layer_ids
        self.first_layer_id = self.kv_cache_layout.first_layer_id

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
            self.block_size, request.all_token_ids, self._seed
        )

        external_block_ids = ucm_block_ids[hbm_hit_block_num:]
        if not external_block_ids:
            return 0, False
        mamba_block_ids: dict[int, bytes] = {}
        try:
            external_hit_blocks = self.store.lookup_on_prefix(external_block_ids) + 1
            if self.kv_cache_layout.use_attn_mamba_hybrid and external_hit_blocks > 0:
                last_hit_attn_block_id = external_block_ids[external_hit_blocks - 1]
                mamba_hit, mamba_block_ids = self._validate_attn_hit_with_mamba(
                    last_hit_attn_block_id
                )
                if not mamba_hit:
                    external_hit_blocks = 0
        except RuntimeError as e:
            external_hit_blocks = 0
            logger.error(f"request {request.request_id} look up error. {e}")
        logger.info_once(
            f"request_id: {request.request_id}, "
            f"total_blocks_num: {len(ucm_block_ids)}, "
            f"hit hbm: {hbm_hit_block_num}, "
            f"hit external: {external_hit_blocks}"
        )
        if self.metrics_config:
            ucmmetrics.update_stats(
                {"interval_lookup_hit_rates": external_hit_blocks / len(ucm_block_ids)},
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
            mamba_block_ids=mamba_block_ids,
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

    def _extend_vllm_block_ids_by_group(
        self, req_meta: RequestMeta, vllm_block_ids_by_group: tuple[list[int], ...]
    ) -> None:
        if not req_meta.vllm_block_ids_by_group:
            req_meta.vllm_block_ids_by_group = tuple(
                [[] for _ in range(len(vllm_block_ids_by_group))]
            )

        for target, source in zip(
            req_meta.vllm_block_ids_by_group, vllm_block_ids_by_group
        ):
            target.extend(source)

    def _generate_dispatch_meta(
        self,
        req_meta: RequestMeta,
        new_tokens: int,
        vllm_block_ids_by_group: tuple[list[int]],
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
        self._extend_vllm_block_ids_by_group(req_meta, vllm_block_ids_by_group)

        n = len(vllm_block_ids_by_group)
        load_pairs: list[tuple[list[bytes], list[int]]] = [([], []) for _ in range(n)]
        dump_pairs: list[tuple[list[bytes], list[int]]] = [([], []) for _ in range(n)]

        group_ids_by_kv_cache_type = self.kv_cache_layout.group_ids_by_kv_cache_type
        attn_group_ids = group_ids_by_kv_cache_type.get(KVCacheType.ATTENTION, [])
        mamba_group_ids = group_ids_by_kv_cache_type.get(KVCacheType.MAMBA, [])

        if need_load:
            for group_id, vllm_block_ids in enumerate(vllm_block_ids_by_group):
                if group_id in attn_group_ids:
                    kernel_block_scale = self.kv_cache_layout.groups[group_id].kernel_block_scale
                    load_ucm_block_ids = ucm_block_ids[
                        hbm_hit_block_num:total_hit_block_num
                    ]
                    load_vllm_block_ids = vllm_block_ids[
                        hbm_hit_block_num:total_hit_block_num
                    ]
                    load_vllm_block_ids = [block_id * kernel_block_scale for block_id in load_vllm_block_ids]
                    load_pairs[group_id] = (load_ucm_block_ids, load_vllm_block_ids)
                elif group_id in mamba_group_ids:
                    mamba_ucm_block_id = req_meta.mamba_block_ids.get(group_id)
                    if mamba_ucm_block_id is None:
                        continue
                    load_pairs[group_id] = (
                        [mamba_ucm_block_id],
                        [vllm_block_ids[-1]],
                    )

        if req_meta.token_processed < req_meta.num_token_ids:
            start_idx = req_meta.token_processed // self.block_size
            end_idx = (req_meta.token_processed + new_tokens) // self.block_size
            attn_dump_ucm_block_ids = ucm_block_ids[start_idx:end_idx]

            mamba_ucm_by_group: dict[int, bytes] = {}
            if attn_dump_ucm_block_ids and self.kv_cache_layout.use_attn_mamba_hybrid:
                last_attn_dump_block_id = attn_dump_ucm_block_ids[-1]
                mamba_ucm_by_group = self._generate_mamba_block_ids(
                    last_attn_dump_block_id
                )

            for group_id, vllm_block_ids in enumerate(vllm_block_ids_by_group):
                if group_id in attn_group_ids:
                    kernel_block_scale = self.kv_cache_layout.groups[group_id].kernel_block_scale
                    dump_vllm_block_ids = req_meta.vllm_block_ids_by_group[group_id][start_idx:end_idx]
                    dump_vllm_block_ids = [block_id * kernel_block_scale for block_id in dump_vllm_block_ids]
                    dump_pairs[group_id] = (
                        attn_dump_ucm_block_ids,
                        dump_vllm_block_ids,
                    )
                elif group_id in mamba_group_ids and attn_dump_ucm_block_ids:
                    mamba_ucm_block_id = mamba_ucm_by_group.get(group_id)
                    if mamba_ucm_block_id is None:
                        continue
                    dump_pairs[group_id] = (
                        [mamba_ucm_block_id],
                        [vllm_block_ids[-1]],
                    )

            req_meta.token_processed += new_tokens

        return RequestDispatchMeta(tuple(load_pairs), tuple(dump_pairs))

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        requests_dispatch_meta = {}
        # for new request, we need to load and dump
        for request in scheduler_output.scheduled_new_reqs:
            request_id = request.req_id
            req_meta = self.requests_meta.get(request_id)
            if req_meta:
                requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request_id],
                    request.block_ids,
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
                    new_block_ids: tuple[list[int], ...] = ()
                    if scheduled_cached_reqs.new_block_ids[i] is not None:
                        new_block_ids = scheduled_cached_reqs.new_block_ids[i]
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
                        request.new_block_ids,
                        request.resumed_from_preemption,
                    )

        # clear finished request
        for request_id in scheduler_output.finished_req_ids:
            self.requests_meta.pop(request_id, None)

        return UCMConnectorMetadata(requests_dispatch_meta)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        request_to_task: dict[str, list[Task]] = defaultdict(list)
        # Blocks that actually entered load_data+wait for this request (for metrics on wait fail).
        request_submitted_blocks: dict[str, int] = defaultdict(int)
        is_load = False
        num_loaded_block = 0
        num_loaded_request = 0
        load_start_time = time.perf_counter() * 1000
        for request_id, request in metadata.request_meta.items():
            for group_id, (load_ucm_ids, load_vllm_ids) in enumerate(
                request.load_block_ids
            ):
                if not load_ucm_ids or not load_vllm_ids:
                    continue
                is_load = True
                num_loaded_block += len(load_ucm_ids)
                num_loaded_request += 1

                if self.tp_rank != 0 and not self.is_mla:
                    for i, ucm_id in enumerate(load_ucm_ids):
                        load_ucm_ids[i] = self.request_hasher(ucm_id)

                total_ptrs = self.kv_cache_layout.extract_block_addrs(
                    load_vllm_ids, group_id
                )
                total_ptrs = total_ptrs.reshape(total_ptrs.shape[0], -1)
                shard_idxs = [0] * len(load_ucm_ids)
                try:
                    task = self.store.load_data(load_ucm_ids, shard_idxs, total_ptrs)
                    request_to_task[request_id].append(task)
                    request_submitted_blocks[request_id] += len(load_ucm_ids)
                except RuntimeError as e:
                    logger.error(f"request {request_id} submit load task error. {e}")
                    self._invalid_block_ids.update(load_vllm_ids)
                    num_loaded_block -= len(load_ucm_ids)

        for request_id, tasks in request_to_task.items():
            try:
                for task in tasks:
                    self.store.wait(task)
            except RuntimeError as e:
                logger.error(f"request {request_id} wait load task error. {e}")
                for _, load_vllm_ids in metadata.request_meta[
                    request_id
                ].load_block_ids:
                    self._invalid_block_ids.update(load_vllm_ids)
                num_loaded_block -= request_submitted_blocks[request_id]

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

        dump_tasks: list[Task] = []
        is_save = False
        num_saved_block = 0
        num_saved_request = 0
        total_ucm_block_ids: dict[int, list[bytes]] = defaultdict(list)
        total_vllm_block_ids: dict[int, list[int]] = defaultdict(list)
        for _, request in metadata.request_meta.items():
            for group_id, (ucm_block_ids, vllm_block_ids) in enumerate(
                request.dump_block_ids
            ):
                if not ucm_block_ids or not vllm_block_ids:
                    continue
                is_save = True
                num_saved_block += len(ucm_block_ids)
                num_saved_request += 1
                total_ucm_block_ids[group_id].extend(ucm_block_ids)
                total_vllm_block_ids[group_id].extend(vllm_block_ids)

        if is_save:
            save_start_time = time.perf_counter() * 1000
            for group_id, ucm_block_ids in total_ucm_block_ids.items():
                if self.tp_rank != 0:
                    for i, ucm_block_id in enumerate(ucm_block_ids):
                        ucm_block_ids[i] = self.request_hasher(ucm_block_id)

                vllm_block_ids = total_vllm_block_ids[group_id]
                total_ptrs = self.kv_cache_layout.extract_block_addrs(
                    vllm_block_ids, group_id
                )
                total_ptrs = total_ptrs.reshape(total_ptrs.shape[0], -1)
                shard_indexs = [0] * len(ucm_block_ids)
                try:
                    event_handle = self._get_dump_event_handle()
                    task = self.store.dump_data(
                        ucm_block_ids, shard_indexs, total_ptrs, event_handle
                    )
                    dump_tasks.append(task)
                except RuntimeError as e:
                    logger.error(f"dump kv cache failed. {e}")
                    return

        if is_save and dump_tasks:
            try:
                for task in dump_tasks:
                    self.store.wait(task)
                save_end_time = time.perf_counter() * 1000
            except RuntimeError as e:
                logger.error(f"wait for dump kv cache failed.{e}")
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

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        # {layer_id: {(request_id, group_id): Task}}
        self.load_tasks: dict[int, dict[tuple[str, int], Task]] = defaultdict(dict)
        self.dump_tasks: dict[str, Task] = {}
        self.dump_total_ptrs_by_group: dict[int, np.ndarray] = {}
        self.use_layerwise = True
        self.kv_cache_layout.use_layerwise = True
        self.is_save = False
        self.need_load = False
        self.layerwise_layers: list[str] = []
        # (request_id, group_id, ucm_block_ids, total_ptrs)
        self.request_data: list[tuple[str, int, list, np.ndarray]] = []
        self._failure_req_ids: set[str] = set()
        logger.info("Init UCMLayerWiseConnector.")

    def _iter_group_block_ids(
        self,
        block_ids: (tuple[tuple[list[bytes], list[int]], ...] | tuple[list[bytes], list[int]]),
    ) -> list[tuple[int, tuple[list[bytes], list[int]]]]:
        if (
            len(block_ids) == 2
            and isinstance(block_ids[0], list)
            and isinstance(block_ids[1], list)
        ):
            return [(self.kv_cache_layout.default_group_id, block_ids)]
        return list(enumerate(block_ids))

    def _hash_block_ids_if_needed(self, block_ids: list[bytes]) -> list[bytes]:
        hashed_block_ids = list(block_ids)
        if self.tp_rank % self.tp_size != 0 and not self.is_mla:
            for i, ucm_block_id in enumerate(hashed_block_ids):
                hashed_block_ids[i] = self.request_hasher(ucm_block_id)
        return hashed_block_ids

    def _get_group_block_id_pair(
        self,
        block_ids: (
            tuple[tuple[list[bytes], list[int]], ...] | tuple[list[bytes], list[int]]
        ),
        group_id: int,
    ) -> tuple[list[bytes], list[int]]:
        for current_group_id, pair in self._iter_group_block_ids(block_ids):
            if current_group_id == group_id:
                return pair
        return [], []

    def _submit_request_load_tasks_for_layer(
        self,
        layer_name: str,
        metadata: "UCMConnectorMetadata",
    ) -> None:
        layer_id = self.layer_name_to_id[layer_name]
        group_id = self.kv_cache_layout.layer_name_to_group_id.get(
            layer_name, self.kv_cache_layout.default_group_id
        )
        local_row = self.kv_cache_layout.groups[group_id].layer_id_to_row[layer_id]

        for request_id, req_group_id, ucm_block_ids, total_ptrs in self.request_data:
            if request_id in self._failure_req_ids or req_group_id != group_id:
                continue
            try:
                shard_indexs = [local_row] * len(ucm_block_ids)
                layer_ptrs = np.ascontiguousarray(total_ptrs[local_row])
                task = self.store.load_data(
                    ucm_block_ids, shard_indexs, layer_ptrs
                )
                self.load_tasks[layer_id][(request_id, req_group_id)] = task
            except RuntimeError as e:
                logger.error(
                    f"request {request_id} group {req_group_id} layer {layer_id} submit load task error. {e}"
                )
                entry = self._get_group_block_id_pair(
                    metadata.request_meta[request_id].load_block_ids, req_group_id
                )
                if entry:
                    self._invalid_block_ids.update(entry[1])
                self._failure_req_ids.add(request_id)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        self.load_tasks.clear()
        self.request_data.clear()
        self._failure_req_ids.clear()
        self.need_load = False
        self.layerwise_layers = sorted(
            self.kv_caches.keys(),
            key=lambda name: self.layer_name_to_id[name],
        )

        for request_id, request in metadata.request_meta.items():
            for group_id, (ucm_block_ids, vllm_block_ids) in self._iter_group_block_ids(
                request.load_block_ids
            ):
                if not ucm_block_ids or not vllm_block_ids:
                    continue

                self.need_load = True
                ucm_block_ids = self._hash_block_ids_if_needed(ucm_block_ids)
                total_ptrs = self.kv_cache_layout.extract_block_addrs(
                    vllm_block_ids, group_id, layer_first=True
                )
                self.request_data.append(
                    (request_id, group_id, ucm_block_ids, total_ptrs)
                )

        if self.need_load and self.layerwise_layers:
            self._submit_request_load_tasks_for_layer(self.layerwise_layers[0], metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self.need_load:
            return
        metadata = self._get_connector_metadata()
        current_layer_id = self.layer_name_to_id[layer_name]

        for (request_id, group_id), task in self.load_tasks.get(
            current_layer_id, {}
        ).items():
            try:
                self.store.wait(task)
            except RuntimeError as e:
                logger.error(
                    f"request {request_id} group {group_id} wait {layer_name} "
                    f"load failed. {e}"
                )
                entry = self._get_group_block_id_pair(
                    metadata.request_meta[request_id].load_block_ids, group_id
                )
                if entry:
                    self._invalid_block_ids.update(entry[1])
                self._failure_req_ids.add(request_id)

        if layer_name not in self.layerwise_layers:
            return
        next_layer_idx = self.layerwise_layers.index(layer_name) + 1
        if next_layer_idx >= len(self.layerwise_layers):
            return
        self._submit_request_load_tasks_for_layer(
            self.layerwise_layers[next_layer_idx], metadata
        )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        # TODO support PP
        if self.is_mla and self.tp_rank % self.tp_size != 0:
            return

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        group_id = self.kv_cache_layout.layer_name_to_group_id.get(
            layer_name, self.kv_cache_layout.default_group_id
        )
        layer_id = self.layer_name_to_id[layer_name]
        local_row = self.kv_cache_layout.groups[group_id].layer_id_to_row[layer_id]

        total_ucm_block_ids, total_vllm_block_ids = [], []
        for _, request in metadata.request_meta.items():
            ucm_block_ids, vllm_block_ids = self._get_group_block_id_pair(
                request.dump_block_ids, group_id
            )
            if not ucm_block_ids or not vllm_block_ids:
                continue

            ucm_block_ids = self._hash_block_ids_if_needed(ucm_block_ids)
            total_ucm_block_ids.extend(ucm_block_ids)
            total_vllm_block_ids.extend(vllm_block_ids)

        if not total_ucm_block_ids:
            return

        self.is_save = True
        if group_id not in self.dump_total_ptrs_by_group:
            self.dump_total_ptrs_by_group[group_id] = (
                self.kv_cache_layout.extract_block_addrs(
                    total_vllm_block_ids, group_id, layer_first=True
                )
            )
        shard_indexs = [local_row] * len(total_ucm_block_ids)
        try:
            ptrs = self.dump_total_ptrs_by_group[group_id]
            layer_ptrs = np.ascontiguousarray(ptrs[local_row])
            event_handle = self._get_dump_event_handle()
            task = self.store.dump_data(
                total_ucm_block_ids, shard_indexs, layer_ptrs, event_handle
            )
            self.dump_tasks[layer_name] = task
        except RuntimeError as e:
            logger.error(f"submit dump task failed. {e}")

    def wait_for_save(self) -> None:
        if not self.is_save:
            return
        try:
            for layer_name in self.kv_caches:
                if layer_name in self.dump_tasks:
                    self.store.wait(self.dump_tasks[layer_name])
        except RuntimeError as e:
            logger.error(f"wait for dump kv cache failed. {e}")
        self.dump_tasks.clear()
        self.is_save = False
        self.dump_total_ptrs_by_group.clear()
        if self.enable_event_sync:
            self.device.destroy_event_handles()


class UCMCPConnector(UCMLayerWiseConnector):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
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
        self.hash_block_size = self.block_size
        self.block_size *= self.cp_world_size
        logger.info("Init UCMCPConnector.")

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

        external_block_ids = ucm_block_ids[hbm_hit_block_num * self.cp_world_size :]
        if not external_block_ids:
            return 0, False
        try:
            external_hit_blocks = self.store.lookup_on_prefix(external_block_ids) + 1
            external_hit_blocks //= self.cp_world_size
        except RuntimeError as e:
            external_hit_blocks = 0
            logger.error(f"request {request.request_id} look up error. {e}")
        logger.info(
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

    def _generate_dispatch_meta(
        self,
        req_meta: RequestMeta,
        new_tokens: int,
        vllm_block_ids: list[int],
        need_load: bool = True,
    ) -> RequestDispatchMeta:
        # Since the block_size on the scheduler side is multiplied by cp_world_size,
        # while the block_size on the UCM side remains unchanged,
        # the selected ucm_blocks need to be expanded by a factor of cp_world_size.
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

        if req_meta.token_processed < req_meta.num_token_ids:
            start_idx = req_meta.token_processed // self.block_size
            end_idx = (req_meta.token_processed + new_tokens) // self.block_size
            dump_ucm_block_ids = ucm_block_ids[
                start_idx * self.cp_world_size : end_idx * self.cp_world_size
            ]
            dump_vllm_block_ids = req_meta.vllm_block_ids[start_idx:end_idx]
            req_meta.token_processed += new_tokens

        return RequestDispatchMeta(
            (load_ucm_block_ids, load_vllm_block_ids),
            (dump_ucm_block_ids, dump_vllm_block_ids),
        )

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

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
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


class UCMConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
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
        if self.launch_config is not None and "hit_ratio" in self.launch_config:
            self.connector = UCMMockConnector(vllm_config, role, kv_cache_config)
        elif (
            hasattr(self._vllm_config.parallel_config, "prefill_context_parallel_size")
            and hasattr(
                self._vllm_config.parallel_config, "decode_context_parallel_size"
            )
            and self._vllm_config.parallel_config.prefill_context_parallel_size
            * self._vllm_config.parallel_config.decode_context_parallel_size
            > 1
        ):
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
