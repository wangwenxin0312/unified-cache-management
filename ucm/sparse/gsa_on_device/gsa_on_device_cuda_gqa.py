"""
CUDA + GQA implementation of GSAOnDevice.
Serves as the base class for the NPU GQA variant.
"""
from typing import List, Optional, Union

import torch

from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext
from vllm.utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput

from ucm.logger import init_logger
from ucm.sparse.base import UcmSparseMetadata, UcmSparseRole
from ucm.sparse.gsa_on_device.gsa_on_device_base import GSAOnDeviceBase
from ucm.sparse.gsa_on_device.hamming_topk import (
    cuda_hamming_topk,
    fake_hamming_topk,
    update_seq_lens,
)
from ucm.sparse.gsa_on_device.hash_encoder import (
    HashEncoder,
    reshape_and_cache_khash_triton,
)

logger = init_logger(__name__)

ReqType = Union[str, int]


class GSAOnDeviceCudaGQA(GSAOnDeviceBase):
    """
    GSAOnDevice for CUDA + GQA (standard multi-head attention) models.
    All methods in this class are CUDA-specific; no NPU branches.
    """

    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        super().__init__(vllm_config, role, is_cuda=True)

        logger.info("GSAOnDevice initialized with GQA model config (CUDA)")

        if role != UcmSparseRole.WORKER:
            return

        # CUDAGraph-compatible pre-allocated buffers (only when not eager)
        if not vllm_config.model_config.enforce_eager:
            device_properties = torch.cuda.get_device_properties(self.device)
            num_sms = device_properties.multi_processor_count
            self.cg_buf_topk_tile_scheduler_metadata = torch.zeros(
                (num_sms, 8), device=self.device, dtype=torch.int32
            )
            self.cg_buf_topk_num_splits = torch.empty(
                (vllm_config.scheduler_config.max_num_seqs + 1),
                device=self.device,
                dtype=torch.int32,
            )
            self.cg_buf_topk_seq_lens = torch.empty(
                (vllm_config.scheduler_config.max_num_seqs + 1),
                device=self.device,
                dtype=torch.int32,
            )

        max_block_per_seq = cdiv(self.max_num_tokens, self.block_size)
        self.full_block_table = torch.zeros(
            (self.max_batch_size, max_block_per_seq),
            dtype=torch.int32,
            device=self.device,
        )
        self.full_seq_lens = torch.zeros(
            (self.max_batch_size,), dtype=torch.int32, device=self.device
        )

        self.head_dim = vllm_config.model_config.get_head_size()
        self.hash_encoder = HashEncoder(
            input_dim=self.head_dim,
            hash_bits=self.head_dim,
            dtype=vllm_config.model_config.dtype,
            device=self.device,
        )

        # GQA decode-phase state
        self.has_decode = False
        self.decode_only = False
        self.topk_block_table = None
        self.topk_seq_lens = None
        self.topk_seq_lens_qwen = None

    # ------------------------------------------------------------------
    # hash_code
    # ------------------------------------------------------------------

    def hash_code(self, query: torch.Tensor) -> torch.Tensor:
        if self.num_q_heads > self.num_key_heads:
            query = query.view(
                query.shape[0],
                self.num_key_heads,
                self.num_q_heads // self.num_key_heads,
                query.shape[2],
            ).mean(2)
        elif self.num_q_heads < self.num_key_heads:
            query = torch.repeat_interleave(
                query, self.num_key_heads // self.num_q_heads, dim=1
            )
        return self.hash_encoder.compute_hash(query).view(torch.bfloat16)

    # ------------------------------------------------------------------
    # cache_k_hash  (called by attention_begin)
    # ------------------------------------------------------------------

    def cache_k_hash(
        self,
        key: torch.Tensor,
        attn_metadata,
        k_hash: torch.Tensor,
        forward_context: ForwardContext,
        layer_name: str,
    ):
        k_hash_compute = self.hash_encoder.compute_hash(key).view(torch.bfloat16)
        valid_k_hash_token = attn_metadata.slot_mapping.flatten().numel()
        reshape_and_cache_khash_triton(
            k_hash_compute[:valid_k_hash_token],
            attn_metadata.slot_mapping.flatten(),
            k_hash,
            block_size=self.block_size,
        )
        if self.has_pc_hit:
            attn = forward_context.no_compile_layers[layer_name]
            kv_cache = attn.kv_cache[forward_context.virtual_engine]
            k_cache = kv_cache[0][0][self.prefix_block_ids]
            k_cache = k_cache.reshape(-1, k_cache.shape[2], k_cache.shape[3])
            prefix_k_hash_compute = self.hash_encoder.compute_hash(k_cache).view(
                torch.bfloat16
            )
            prefix_valid_k_hash_token = self.prefix_slot_mapping.flatten().numel()
            reshape_and_cache_khash_triton(
                prefix_k_hash_compute[:prefix_valid_k_hash_token],
                self.prefix_slot_mapping.flatten(),
                k_hash,
                block_size=self.block_size,
            )

    # ------------------------------------------------------------------
    # update_decode_topk  (called by attention_begin for decode requests)
    # ------------------------------------------------------------------

    def update_decode_topk(
        self,
        query: torch.Tensor,
        k_hash: torch.Tensor,
        attn_metadata,
    ):
        q_hash = self.hash_code(query=query[: self.num_reqs])

        block_table_decode = cuda_hamming_topk(
            q_hash.unsqueeze(1),
            k_hash,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            topk_token=self.hash_topk_tokens,
            sink_token=self.block_size,
            recent_token=self.block_size * 4,
            is_mla=False,
        )
        topk = block_table_decode.shape[1]
        if self.decode_only:
            self.new_block_table[: self.num_reqs, :topk] = block_table_decode
            self.new_block_table[: self.num_reqs, topk:] = 0
            attn_metadata.block_table = self.new_block_table
            self.new_seq_lens[: self.num_reqs] = self.topk_seq_lens_qwen
            attn_metadata.seq_lens = self.new_seq_lens
        else:
            self.new_block_table[self.decode_req_ids, :topk] = block_table_decode[
                self.decode_req_ids, :topk
            ]
            self.new_block_table[self.decode_req_ids, topk:] = 0
            attn_metadata.block_table = self.new_block_table
            self.new_seq_lens.index_copy_(
                0, self.decode_req_ids, self.topk_seq_lens_qwen[self.decode_req_ids]
            )
            attn_metadata.seq_lens = self.new_seq_lens

        self.topk_block_table = attn_metadata.block_table
        self.topk_seq_lens = attn_metadata.seq_lens

    # ------------------------------------------------------------------
    # apply_skip_layer: write cached topk results into attn_metadata
    # ------------------------------------------------------------------

    def _apply_skip_layer(self, attn_metadata):
        attn_metadata.block_table = self.topk_block_table
        attn_metadata.seq_lens = self.topk_seq_lens

    # ------------------------------------------------------------------
    # attention_begin / attention_finished
    # ------------------------------------------------------------------

    def attention_begin(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
        output: Optional[torch.Tensor] = None,
        phase: Optional[str] = None,
        k_hash: Optional[torch.Tensor] = None,
        decode_ql_nope: Optional[torch.Tensor] = None,
        decode_q_pe: Optional[torch.Tensor] = None,
    ):
        attn_metadata = self.get_layer_attn_metadata(forward_context, layer_name)
        is_rollback_layer, is_skip_hash_layer = self.get_layer_state(layer_name)

        if not is_rollback_layer and not is_skip_hash_layer:
            self.cache_k_hash(key, attn_metadata, k_hash, forward_context, layer_name)

        if not self.gsa_enabled:
            return query, key, value, output

        if self.has_decode:
            if not is_rollback_layer:
                if is_skip_hash_layer:
                    self._apply_skip_layer(attn_metadata)
                else:
                    self.update_decode_topk(query, k_hash, attn_metadata)

        return query, key, value, output

    def attention_finished(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_output: torch.Tensor,
        layer_name: str,
        forward_context: ForwardContext,
        phase: Optional[str] = None,
    ) -> None:
        if not self.gsa_enabled:
            return
        if self.has_decode:
            is_rollback_layer, _ = self.get_layer_state(layer_name)
            if not is_rollback_layer:
                attn_metadata = self.get_layer_attn_metadata(forward_context, layer_name)
                attn_metadata.block_table = self.ori_block_table_decode
                attn_metadata.seq_lens = self.ori_seq_lens_decode

    # ------------------------------------------------------------------
    # initialize_kv_hash_cache_tensors
    # ------------------------------------------------------------------

    def initialize_kv_hash_cache_tensors(self, kv_caches, device):
        dtype = torch.bfloat16
        for layer_name, kv_cache in kv_caches.items():
            is_rollback_layer, is_skip_hash_layer = self.get_layer_state(layer_name)
            if not is_rollback_layer and not is_skip_hash_layer:
                khash_cache_shape = list(kv_cache[0].shape)
                khash_cache_shape[-1] //= dtype.itemsize * 8
                khash_cache = torch.zeros(khash_cache_shape, dtype=dtype, device=device)
            else:
                khash_cache = None
            kv_caches[layer_name] = (kv_cache, khash_cache)

    # ------------------------------------------------------------------
    # get_block_table_row / get_seq_lens
    # ------------------------------------------------------------------

    def get_block_table_row(self, attn_metadata, req_row_id, prefill_row_id):
        return attn_metadata.block_table[req_row_id]

    def get_seq_lens(self, attn_metadata):
        return getattr(attn_metadata, "seq_lens", None)

    # ------------------------------------------------------------------
    # prepare_cuda_decode_sparse_meta (CUDA GQA only)
    # ------------------------------------------------------------------

    def prepare_cuda_decode_sparse_meta(self, attn_metadata, num_decodes: int):
        self.full_seq_lens[: self.num_reqs].copy_(attn_metadata.seq_lens, True)
        self.full_block_table[: self.num_reqs].copy_(attn_metadata.block_table, True)

        self.ori_seq_lens_decode = self.full_seq_lens[: self.num_reqs]
        self.ori_block_table_decode = self.full_block_table[: self.num_reqs]

        if not self.decode_only:
            self.decode_req_ids_buf.copy_to_gpu(num_decodes)
            self.decode_req_ids = self.decode_req_ids_buf.gpu[:num_decodes]

        self.topk_seq_lens_qwen = update_seq_lens(
            attn_metadata.seq_lens,
            topk_token=self.hash_topk_tokens,
            block_size=self.block_size,
        )
        self.new_block_table = attn_metadata.block_table
        self.new_seq_lens = attn_metadata.seq_lens

    def _build_decode_sparse_meta(self, attn_metadata, num_decodes: int):
        self.has_decode = num_decodes > 0
        self.decode_only = self.has_decode and (num_decodes == self.num_reqs)
        if self.has_decode:
            self.prepare_cuda_decode_sparse_meta(attn_metadata, num_decodes)

    # ------------------------------------------------------------------
    # build_sparse_meta
    # ------------------------------------------------------------------

    def build_sparse_meta(
        self, scheduler_output, requests, input_batch, attn_metadata
    ) -> UcmSparseMetadata:
        self.num_reqs = len(scheduler_output.num_scheduled_tokens)

        if isinstance(attn_metadata, dict):
            attn_metadata = next(iter(attn_metadata.values()))

        seq_lens = self.get_seq_lens(attn_metadata)
        if seq_lens is None:
            return

        num_long_reqs = int(
            (seq_lens[: self.num_reqs] >= self.seq_len_threshold).sum().item()
        )
        self.gsa_enabled = num_long_reqs >= self.concurrency_threshold

        num_decodes, _, _, _ = self._run_build_sparse_meta_loop(
            scheduler_output, requests, input_batch, attn_metadata
        )

        if not self.gsa_enabled:
            return

        self._build_decode_sparse_meta(attn_metadata, num_decodes)

    # ------------------------------------------------------------------
    # execute_begin / execute_finished
    # ------------------------------------------------------------------

    def execute_begin(self, scheduler_output: SchedulerOutput):
        pass  # no per-step reset needed on CUDA GQA

    def execute_finished(self, logits_indices: torch.Tensor):
        self.has_decode = False
        self.gsa_enabled = False
        self.decode_only = False
        self.has_pc_hit = False
        return logits_indices

    # ------------------------------------------------------------------
    # maybe_init_cudagraph_buffers_for_topk  (CUDA only helper)
    # ------------------------------------------------------------------

    def maybe_init_cudagraph_buffers_for_topk(
        self,
        n,
        tile_scheduler_metadata,
        topk_tile_scheduler_metadata,
        topk_num_splits,
        topk_seq_lens,
    ):
        sm_parts = tile_scheduler_metadata.size(0)
        topk_tile_scheduler_metadata_view = self.cg_buf_topk_tile_scheduler_metadata[
            :sm_parts
        ]
        topk_tile_scheduler_metadata_view.copy_(topk_tile_scheduler_metadata)
        topk_tile_scheduler_metadata = topk_tile_scheduler_metadata_view

        topk_num_splits_view = self.cg_buf_topk_num_splits[:n]
        topk_num_splits_view.copy_(topk_num_splits)
        self.cg_buf_topk_num_splits[n:].fill_(topk_num_splits[-1])
        topk_num_splits = topk_num_splits_view

        topk_seq_lens_view = self.cg_buf_topk_seq_lens[: topk_seq_lens.size(0)]
        topk_seq_lens_view.copy_(topk_seq_lens)
        topk_seq_lens = topk_seq_lens_view
        return topk_tile_scheduler_metadata, topk_num_splits, topk_seq_lens
