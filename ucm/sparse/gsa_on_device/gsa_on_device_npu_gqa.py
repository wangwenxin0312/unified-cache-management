"""
NPU + GQA implementation of GSAOnDevice.
Inherits from GSAOnDeviceCudaGQA and overrides NPU-specific methods.
"""
from typing import List, Optional, Union

import torch
import torch_npu
import ucm_custom_ops
from vllm_ascend.attention.attention_v1 import AscendAttentionState

from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext
from vllm.utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput

from ucm.logger import init_logger
from ucm.sparse.base import UcmSparseMetadata, UcmSparseRole
from ucm.sparse.gsa_on_device.gsa_on_device_cuda_gqa import GSAOnDeviceCudaGQA
from ucm.sparse.gsa_on_device.hamming_topk import update_seq_lens

logger = init_logger(__name__)

ReqType = Union[str, int]


class GSAOnDeviceNpuGQA(GSAOnDeviceCudaGQA):
    """
    GSAOnDevice for NPU + GQA models.
    Overrides all CUDA-specific methods from GSAOnDeviceCudaGQA with NPU equivalents.
    """

    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        # Call GSAOnDeviceBase.__init__ directly, bypassing CUDA-specific GQA init,
        # then run the NPU+GQA-specific initialization.
        from ucm.sparse.gsa_on_device.gsa_on_device_base import GSAOnDeviceBase
        GSAOnDeviceBase.__init__(self, vllm_config, role, is_cuda=False)

        logger.info("GSAOnDevice initialized with GQA model config (NPU)")

        if role != UcmSparseRole.WORKER:
            return

        self.head_dim = vllm_config.model_config.get_head_size()
        from ucm.sparse.gsa_on_device.hash_encoder import HashEncoder
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

        # NPU GQA specific
        self.decode_mask = None
        self.decode_mask_npu = None
        self.is_tensor_computed = False
        self.hamming_keep_chunks_head = 1
        self.hamming_keep_chunks_tail = 4

        self.chunk_sizes_for_hamming_full = torch.full(
            [self.max_batch_size],
            fill_value=self.block_size,
            dtype=torch.int32,
            device=self.device,
        )
        self.topk_for_hamming_full = torch.full(
            [self.max_batch_size],
            fill_value=self.hash_topk_tokens // self.block_size,
            dtype=torch.int32,
            device=self.device,
        )
        self.topk_for_hamming_full_cpu = torch.full(
            [self.max_batch_size],
            fill_value=self.hash_topk_tokens // self.block_size,
            dtype=torch.int32,
            device="cpu",
        )
        self.seq_lens_for_hamming = torch.zeros(
            [self.max_batch_size], dtype=torch.int32, device=self.device
        )
        self.hamming_output = torch.zeros(
            [
                self.max_batch_size,
                self.num_key_heads,
                cdiv(vllm_config.model_config.max_model_len, self.block_size),
            ],
            dtype=torch.int32,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # init_for_pc: NPU uses int32 slot mapping + prefix_token_len_full
    # ------------------------------------------------------------------

    def init_for_pc(self):
        self.prefix_slot_mapping_buf = torch.empty(
            self.max_num_tokens * self.max_batch_size,
            device=self.device,
            dtype=torch.int32,
        )
        self.prefix_token_len_full = torch.zeros(
            (self.max_batch_size,), dtype=torch.int32, device=self.device
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
    # Template hooks for prefix token length tracking (NPU only)
    # ------------------------------------------------------------------

    def _record_prefix_token_len(self, num_pc_hit: int, num_prefix_tokens: int):
        self.prefix_token_len_full[num_pc_hit] = num_prefix_tokens

    def _finalize_prefix_token_len(self, num_pc_hit: int):
        self.prefix_token_len = self.prefix_token_len_full[:num_pc_hit]

    # ------------------------------------------------------------------
    # cache_k_hash  (NPU version)
    # ------------------------------------------------------------------

    def cache_k_hash(
        self,
        key: torch.Tensor,
        attn_metadata,
        k_hash: torch.Tensor,
        forward_context: ForwardContext,
        layer_name: str,
    ):
        if not self.is_tensor_computed:
            if self.decode_mask.any():
                if self.slice_enabled:
                    self.batch_size_for_hamming = self.num_decode_requests
                else:
                    self.batch_size_for_hamming = len(attn_metadata.query_lens)
                    self.decode_mask_npu = (attn_metadata.query_lens_device == 1) & (
                        attn_metadata.seq_lens_device[
                            : attn_metadata.query_lens_device.numel()
                        ]
                        >= self.seq_len_threshold
                    )
                self.topk_for_hamming = self.topk_for_hamming_full[
                    : self.batch_size_for_hamming
                ]
                self.chunk_sizes_for_hamming = self.chunk_sizes_for_hamming_full[
                    : self.batch_size_for_hamming
                ]
                self.seq_lens_for_hamming = attn_metadata.seq_lens_device[
                    : self.batch_size_for_hamming
                ]
                self.max_seq_len_for_hamming = torch.max(
                    attn_metadata.seq_lens[: self.batch_size_for_hamming]
                ).item()
                self.block_table_decode = self.ori_block_table_decode[
                    : self.batch_size_for_hamming
                ]
                self.new_block_tables = attn_metadata.block_tables
                self.is_tensor_computed = True

        k_hash_compute = self.hash_encoder.compute_hash(key)
        k_hash_compute = (
            k_hash_compute.transpose(0, 1)
            .reshape(-1, k_hash_compute.shape[-1])
            .contiguous()
        )
        torch.ops._C_ucm.npu_reshape_and_cache_bnsd(
            k_hash_compute,
            k_hash,
            attn_metadata.slot_mapping,
            attn_metadata.query_lens_device,
            k_hash,
        )
        if self.has_pc_hit:
            attn = forward_context.no_compile_layers[layer_name]
            kv_cache = attn.kv_cache[forward_context.virtual_engine]
            k_cache = kv_cache[0][0][self.prefix_block_ids]
            k_cache = k_cache.reshape(-1, k_cache.shape[2], k_cache.shape[3])
            prefix_k_hash_compute = self.hash_encoder.compute_hash(k_cache)
            prefix_k_hash_compute = (
                prefix_k_hash_compute.transpose(0, 1)
                .reshape(-1, prefix_k_hash_compute.shape[-1])
                .contiguous()
            )
            torch.ops._C_ucm.npu_reshape_and_cache_bnsd(
                prefix_k_hash_compute,
                k_hash,
                self.prefix_slot_mapping.flatten(),
                self.prefix_token_len,
                k_hash,
            )

    # ------------------------------------------------------------------
    # update_decode_topk  (NPU version)
    # ------------------------------------------------------------------

    def update_decode_topk(
        self,
        query: torch.Tensor,
        k_hash: torch.Tensor,
        attn_metadata,
    ):
        q_start = attn_metadata.query_start_loc[: self.batch_size_for_hamming + 1]
        if self.slice_enabled:
            q_decode = query[: self.batch_size_for_hamming]
        else:
            q_decode = query.index_select(0, q_start[:-1])
        q_hash = self.hash_encoder.compute_hash(q_decode).unsqueeze(2).contiguous()

        torch.ops._C_ucm.npu_hamming_dist_top_k(
            q_hash,
            k_hash,
            None,
            self.topk_for_hamming,
            self.seq_lens_for_hamming,
            self.chunk_sizes_for_hamming,
            self.max_seq_len_for_hamming,
            self.hamming_keep_chunks_head,
            self.hamming_keep_chunks_tail,
            0,  # support_offload disabled
            self.block_table_decode,
            (self.decode_mask_npu if not self.slice_enabled else None),
            self.hamming_output[: self.batch_size_for_hamming],
        )
        new_seq_lens = self.topk_seq_lens_qwen
        attn_metadata.seq_lens = new_seq_lens

        self.new_block_tables[: self.batch_size_for_hamming] = self.hamming_output[
            : self.batch_size_for_hamming, 0, :
        ]
        attn_metadata.block_tables = self.new_block_tables

        self.topk_block_table = attn_metadata.block_tables
        self.topk_seq_lens = attn_metadata.seq_lens

    # ------------------------------------------------------------------
    # _apply_skip_layer: NPU uses block_tables (plural)
    # ------------------------------------------------------------------

    def _apply_skip_layer(self, attn_metadata):
        attn_metadata.block_tables = self.topk_block_table
        attn_metadata.seq_lens = self.topk_seq_lens

    # ------------------------------------------------------------------
    # attention_finished: NPU uses block_tables
    # ------------------------------------------------------------------

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
                attn_metadata.block_tables = self.ori_block_table_decode
                attn_metadata.seq_lens = self.ori_seq_lens_decode

    # ------------------------------------------------------------------
    # initialize_kv_hash_cache_tensors (NPU GQA version)
    # ------------------------------------------------------------------

    def initialize_kv_hash_cache_tensors(self, kv_caches, device):
        for layer_name, kv_cache in kv_caches.items():
            is_rollback_layer, is_skip_hash_layer = self.get_layer_state(layer_name)
            k_cache_shape = kv_cache[0].shape
            khash_cache_shape = (
                k_cache_shape[0],
                k_cache_shape[2],
                k_cache_shape[1],
                self.hash_encoder.hash_bits // 8,
            )
            if not is_rollback_layer and not is_skip_hash_layer:
                khash_cache = torch.empty(
                    khash_cache_shape, dtype=torch.uint8, device=device
                )
            else:
                khash_cache = None
            kv_caches[layer_name] = (kv_cache, khash_cache)

    # ------------------------------------------------------------------
    # get_block_table_row: NPU uses block_tables
    # ------------------------------------------------------------------

    def get_block_table_row(self, attn_metadata, req_row_id, prefill_row_id):
        return attn_metadata.block_tables[req_row_id]

    # ------------------------------------------------------------------
    # build_decode_attention_meta_npu (called externally by integration layer)
    # ------------------------------------------------------------------

    def build_decode_attention_meta_npu(self, query_lens, seq_lens, block_table):
        self.ori_seq_lens_decode = seq_lens.clone()
        self.ori_block_table_decode = block_table.clone()

        self.decode_mask = (query_lens == 1) & (
            seq_lens[: query_lens.numel()] >= self.seq_len_threshold
        )
        self.num_decode_requests = self.decode_mask.sum().item()
        if self.num_decode_requests > 0:
            self.slice_enabled = (
                self.decode_mask[: self.num_decode_requests].all().item()
            )
        else:
            self.slice_enabled = False

        if self.decode_mask.any():
            self.topk_seq_lens_qwen = update_seq_lens(
                seq_lens,
                topk_token=self.hash_topk_tokens,
                block_size=self.block_size,
            )
            self.topk_seq_lens_qwen[: query_lens.numel()][~self.decode_mask] = seq_lens[
                : query_lens.numel()
            ][~self.decode_mask]

    # ------------------------------------------------------------------
    # _build_decode_sparse_meta override: no CUDA prepare on NPU
    # ------------------------------------------------------------------

    def _build_decode_sparse_meta(self, attn_metadata, num_decodes: int):
        self.has_decode = num_decodes > 0
        self.decode_only = self.has_decode and (num_decodes == self.num_reqs)
        # NPU GQA decode metadata is prepared externally via build_decode_attention_meta_npu

    # ------------------------------------------------------------------
    # execute_begin: reset is_tensor_computed each step (NPU only)
    # ------------------------------------------------------------------

    def execute_begin(self, scheduler_output: SchedulerOutput):
        self.is_tensor_computed = False
