from importlib import resources
from pathlib import Path
from typing import List, Optional, Union

import torch

from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext
from vllm.utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from ucm.logger import init_logger
from ucm.sparse.base import (
    INVALID_SLOT,
    UcmSparseBase,
    UcmSparseCpuGpuBuffer,
    UcmSparseMetadata,
    UcmSparseRole,
)
from ucm.sparse.gsa_on_device.gsa_on_device_config import GSAOnDeviceConfig

logger = init_logger(__name__)

ReqType = Union[str, int]


def gsa_on_device_config_path_for_model(vllm_config) -> str:
    model = vllm_config.model_config.model.lower()
    logger.info(f"[GSAOnDevice] model name: {model}")

    if "deepseek" in model and "r1" in model:
        rel = "ucm/sparse/gsa_on_device/configs/gsa_on_device_deepseek_r1_awq_config.json"
    elif "qwen3" in model and "32b" in model and "coder" not in model:
        rel = "ucm/sparse/gsa_on_device/configs/gsa_on_device_qwen3_32B_config.json"
    elif "qwen3" in model and "30b" in model and "coder" in model:
        rel = "ucm/sparse/gsa_on_device/configs/gsa_on_device_qwen3_coder_30B_A3B_config.json"
    elif "qwen3" in model and "4b" in model:
        rel = "ucm/sparse/gsa_on_device/configs/gsa_on_device_qwen3_4B_config.json"
    elif "qwq" in model and "32b" in model:
        rel = "ucm/sparse/gsa_on_device/configs/gsa_on_device_qwq_32B_config.json"
    elif "deepseek" in model and "v2" in model:
        rel = "ucm/sparse/gsa_on_device/configs/gsa_on_device_deepseek_v2_lite_config.json"
    else:
        raise ValueError(f"[GSAOnDevice] Unsupported model for gsa_on_device: {model}")

    logger.info(f"[GSAOnDevice] target relative path: {rel}")

    cur = Path(__file__).resolve()
    repo = cur
    for depth in range(30):
        if (
            (repo / "pyproject.toml").is_file()
            or (repo / "setup.cfg").is_file()
            or (repo / ".git").exists()
        ):
            p = repo / rel
            logger.info(f"[GSAOnDevice] repo root detected at depth={depth}: {repo}")
            if p.is_file():
                logger.info(f"[GSAOnDevice] config loaded from SOURCE tree: {p}")
                return str(p)
            logger.warning(f"[GSAOnDevice] repo root found but config missing: {p}")
            break
        if repo.parent == repo:
            logger.debug("[GSAOnDevice] reached filesystem root, stop searching")
            break
        repo = repo.parent

    sub = rel[len("ucm/"):] if rel.startswith("ucm/") else rel
    res = resources.files("ucm").joinpath(*sub.split("/"))
    with resources.as_file(res) as p:
        logger.info(f"[GSAOnDevice] config loaded from PACKAGE resource (wheel): {p}")
        return str(p)


class GSAOnDeviceBase(UcmSparseBase):
    """
    Common base for all GSAOnDevice variants.
    Subclasses are split by model type (GQA / MLA) and platform (CUDA / NPU).
    """

    is_cuda: bool  # set by subclass __init__

    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole, is_cuda: bool):
        super().__init__(vllm_config, role)
        self.rank = vllm_config.parallel_config.rank
        self.is_cuda = is_cuda
        self.gsa_enabled = False

        self.device = torch.device(
            f"{'cuda' if is_cuda else 'npu'}:{self.rank}"
        )

        self.num_q_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_key_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.block_size = vllm_config.cache_config.block_size

        gsa_on_device_config_path = gsa_on_device_config_path_for_model(vllm_config)
        self.gsa_on_device_config = GSAOnDeviceConfig.from_json(gsa_on_device_config_path)
        logger.info(f"read gsa_on_device config file : {gsa_on_device_config_path} ")

        self.hash_topk_tokens = self.gsa_on_device_config.vllm_hash_attention_topk
        self.hash_rollback_layers = self.gsa_on_device_config.vllm_hash_attention_rollback_layers
        self.hash_skip_layers = self.gsa_on_device_config.vllm_hash_attention_skip_layers

        if is_cuda:
            self.seq_len_threshold = self.gsa_on_device_config.gpu_seq_len_threshold
            self.concurrency_threshold = self.gsa_on_device_config.gpu_concurrency_threshold
        else:
            self.seq_len_threshold = self.gsa_on_device_config.npu_seq_len_threshold
            self.concurrency_threshold = self.gsa_on_device_config.npu_concurrency_threshold

        assert (
            self.seq_len_threshold >= self.gsa_on_device_config.vllm_hash_attention_topk
        ), "seq_len_threshold must be larger than or equal to vllm_hash_attention_topk"
        assert (
            self.gsa_on_device_config.vllm_hash_attention_topk % self.block_size == 0
        ), "vllm_hash_attention_topk must be divisible by block_size"
        assert (
            self.gsa_on_device_config.vllm_hash_attention_topk
            <= vllm_config.model_config.max_model_len
        ), "vllm_hash_attention_topk must be less than max_model_len"

        if role != UcmSparseRole.WORKER:
            return

        self.ori_seq_lens_decode = None
        self.ori_block_table_decode = None
        self.has_pc_hit = False
        self.is_prefill_flag: dict[str, bool] = dict()
        self._k_scale = torch.tensor(1.0, dtype=torch.float32)

        self.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.decode_req_ids_buf = self._make_buffer(self.max_batch_size, dtype=torch.int64)
        self.max_num_tokens = vllm_config.model_config.max_model_len
        self.init_for_pc()

    # ------------------------------------------------------------------
    # Platform hook: CUDA uses int64 slot mapping; NPU subclasses override
    # ------------------------------------------------------------------

    def init_for_pc(self):
        self.prefix_slot_mapping_buf = torch.empty(
            self.max_num_tokens * self.max_batch_size,
            device=self.device,
            dtype=torch.int64,
        )
        self.prefix_block_ids_buf = torch.empty(
            cdiv(self.max_num_tokens * self.max_batch_size, self.block_size),
            device=self.device,
            dtype=torch.int32,
        )
        self.token_idx_buf = torch.arange(
            self.max_num_tokens, device=self.device, dtype=torch.int64
        )

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _make_buffer(
        self, *size: Union[int, torch.SymInt], dtype: torch.dtype, numpy: bool = True
    ) -> UcmSparseCpuGpuBuffer:
        return UcmSparseCpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=True, with_numpy=numpy
        )

    def get_layer_attn_metadata(self, forward_context: ForwardContext, layer_name: str):
        attn_meta = forward_context.attn_metadata
        return attn_meta[layer_name] if isinstance(attn_meta, dict) else attn_meta

    def get_layer_state(self, layer_name: str):
        layer_id = int(layer_name.split(".")[2])
        is_rollback_layer = layer_id in self.hash_rollback_layers
        is_skip_hash_layer = (
            layer_id < len(self.hash_skip_layers) and self.hash_skip_layers[layer_id]
        )
        return is_rollback_layer, is_skip_hash_layer

    def rebuild_prefix_cache_info_for_req(
        self,
        block_table_row: torch.Tensor,
        num_prompt_tokens: int,
        qlen: int,
        block_size: int,
    ):
        assert 0 <= qlen <= num_prompt_tokens
        num_prefix_tokens = num_prompt_tokens - qlen
        if num_prefix_tokens <= 0:
            empty = block_table_row[:0]
            return 0, 0, empty, empty

        num_prefix_blocks = (num_prefix_tokens + block_size - 1) // block_size
        prefix_block_ids = block_table_row[:num_prefix_blocks]

        token_idx = self.token_idx_buf[:num_prefix_tokens]
        block_indices = token_idx // block_size
        block_offsets = token_idx - block_indices * block_size
        token_block_numbers = prefix_block_ids.index_select(0, block_indices)

        prefix_slot_mapping = token_block_numbers * block_size + block_offsets
        return num_prefix_tokens, num_prefix_blocks, prefix_block_ids, prefix_slot_mapping

    # ------------------------------------------------------------------
    # Template hooks for build_sparse_meta (implemented in subclasses)
    # ------------------------------------------------------------------

    def _record_prefix_token_len(self, num_pc_hit: int, num_prefix_tokens: int):
        """NPU subclasses store per-request prefix token length; CUDA is a no-op."""
        pass

    def _finalize_prefix_token_len(self, num_pc_hit: int):
        """NPU subclasses slice the prefix_token_len buffer; CUDA is a no-op."""
        pass

    def _build_decode_sparse_meta(self, attn_metadata, num_decodes: int):
        """GQA CUDA builds decode sparse metadata here; other variants override or no-op."""
        pass

    # ------------------------------------------------------------------
    # build_sparse_meta: common request-loop logic shared by all variants
    # ------------------------------------------------------------------

    def _run_build_sparse_meta_loop(
        self, scheduler_output, requests, input_batch, attn_metadata
    ):
        """
        Shared core loop used by all build_sparse_meta implementations.
        Returns (num_decodes, num_pc_hit, all_prefix_tokens, all_prefix_blocks).
        Also populates has_pc_hit, prefix_slot_mapping, prefix_block_ids.
        """
        compute_q_lens = (
            attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
        )
        self.decode_req_ids_buf.clear()

        num_decodes = 0
        num_pc_hit = 0
        all_prefix_tokens = 0
        all_prefix_blocks = 0
        prefill_row_id = 0

        for req_id, num_scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
            req = requests[req_id]
            is_decode = req_id in self.is_prefill_flag and not self.is_prefill_flag[req_id]
            is_first_prefil = req_id not in self.is_prefill_flag
            is_prefill = is_first_prefil or self.is_prefill_flag[req_id]
            is_last_chunk = is_prefill and (
                req.num_computed_tokens + num_scheduled_tokens >= req.num_prompt_tokens
            )

            if is_decode:
                self.decode_req_ids_buf.np[num_decodes] = input_batch.req_id_to_index[req_id]
                num_decodes += 1

            if is_first_prefil:
                self.is_prefill_flag[req_id] = True
                req_row_id = input_batch.req_id_to_index[req_id]
                ext_tokens = int(
                    scheduler_output.num_external_computed_tokens_per_req.get(req_id, 0)
                )

                if ext_tokens > 0:
                    block_table_row = self.get_block_table_row(
                        attn_metadata, req_row_id, prefill_row_id
                    )
                    (
                        num_prefix_tokens,
                        num_prefix_blocks,
                        prefix_block_ids,
                        prefix_slot_mapping,
                    ) = self.rebuild_prefix_cache_info_for_req(
                        block_table_row=block_table_row,
                        num_prompt_tokens=req.num_prompt_tokens,
                        qlen=compute_q_lens[req_row_id],
                        block_size=self.block_size,
                    )
                    self.prefix_slot_mapping_buf[
                        all_prefix_tokens: all_prefix_tokens + num_prefix_tokens
                    ] = prefix_slot_mapping
                    self.prefix_block_ids_buf[
                        all_prefix_blocks: all_prefix_blocks + num_prefix_blocks
                    ] = prefix_block_ids
                    self._record_prefix_token_len(num_pc_hit, num_prefix_tokens)
                    all_prefix_tokens += num_prefix_tokens
                    all_prefix_blocks += num_prefix_blocks
                    num_pc_hit += 1
                    prefill_row_id += 1

            if is_last_chunk:
                self.is_prefill_flag[req_id] = False

        self.has_pc_hit = num_pc_hit > 0
        if self.has_pc_hit:
            self.prefix_slot_mapping = self.prefix_slot_mapping_buf[:all_prefix_tokens]
            self.prefix_block_ids = self.prefix_block_ids_buf[:all_prefix_blocks]
            self._finalize_prefix_token_len(num_pc_hit)

        return num_decodes, num_pc_hit, all_prefix_tokens, all_prefix_blocks

    # ------------------------------------------------------------------
    # UcmSparseBase interface stubs (overridden in subclasses)
    # ------------------------------------------------------------------

    def get_block_table_row(self, attn_metadata, req_row_id, prefill_row_id):
        raise NotImplementedError

    def get_seq_lens(self, attn_metadata):
        raise NotImplementedError

    def request_begin(self, request_id: ReqType, prompt_token_ids: List[int]):
        pass

    def request_finished_in_scheduler(self, request_id: Union[int, str]):
        pass

    def estimate_num_slots_sparsed(self, request: Request) -> int:
        return INVALID_SLOT

    def _free_cached_request(self, request_id: Union[int, str]) -> None:
        if request_id not in self.is_prefill_flag:
            return
        del self.is_prefill_flag[request_id]

    def update_states(self, scheduler_output: SchedulerOutput) -> None:
        for req_id in scheduler_output.finished_req_ids:
            self._free_cached_request(req_id)

        req_data = scheduler_output.scheduled_cached_reqs
        for req_id, resumed_from_preemption in zip(
            req_data.req_ids, req_data.resumed_from_preemption
        ):
            if resumed_from_preemption:
                self._free_cached_request(req_id)
