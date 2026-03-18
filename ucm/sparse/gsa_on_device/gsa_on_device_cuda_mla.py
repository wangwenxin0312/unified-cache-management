"""
CUDA + MLA (DeepSeek Multi-Latent Attention) implementation of GSAOnDevice.
Serves as the base class for the NPU MLA variant.
"""
from typing import List, Optional, Union

import torch

from vllm import _custom_ops as ops
from vllm.attention.ops.flashmla import get_mla_metadata
from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext
from vllm.v1.core.sched.output import SchedulerOutput

from ucm.logger import init_logger
from ucm.sparse.base import UcmSparseMetadata, UcmSparseRole
from ucm.sparse.gsa_on_device.gsa_on_device_base import GSAOnDeviceBase
from ucm.sparse.gsa_on_device.hamming_topk import (
    cuda_hamming_topk,
    update_seq_lens,
)
from ucm.sparse.gsa_on_device.hash_encoder import HashEncoder

logger = init_logger(__name__)

ReqType = Union[str, int]


class GSAOnDeviceCudaMLA(GSAOnDeviceBase):
    """
    GSAOnDevice for CUDA + MLA (DeepSeek) models.
    All methods in this class are CUDA-specific; no NPU branches.
    """

    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        super().__init__(vllm_config, role, is_cuda=True)

        logger.info("GSAOnDevice initialized with MLA model config (CUDA)")

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

        self.hash_reduction_head_num = (
            self.gsa_on_device_config.vllm_hash_attention_reduction_head_num
        )
        self.kv_lora_rank = getattr(
            vllm_config.model_config.hf_text_config, "kv_lora_rank", None
        )
        self.qk_rope_head_dim = getattr(
            vllm_config.model_config.hf_text_config, "qk_rope_head_dim", None
        )
        self.hash_encoder_nope = HashEncoder(
            input_dim=self.kv_lora_rank,
            hash_bits=self.kv_lora_rank,
            dtype=vllm_config.model_config.dtype,
            device=self.device,
        )
        self.hash_encoder_rope = HashEncoder(
            input_dim=self.qk_rope_head_dim,
            hash_bits=self.qk_rope_head_dim,
            dtype=vllm_config.model_config.dtype,
            device=self.device,
        )

        # MLA decode-phase state
        self.origin_tile_scheduler_metadata = None
        self.origin_num_splits = None

    # ------------------------------------------------------------------
    # hash_code
    # ------------------------------------------------------------------

    def hash_code(
        self,
        nope: torch.Tensor,
        rope: torch.Tensor,
        reduction_head_num: int = 1,
    ):
        if reduction_head_num > 1:
            nope = nope.view(
                nope.shape[0],
                reduction_head_num,
                nope.shape[1] // reduction_head_num,
                nope.shape[2],
            ).mean(dim=1)
            rope = rope.view(
                rope.shape[0],
                reduction_head_num,
                rope.shape[1] // reduction_head_num,
                rope.shape[2],
            ).mean(dim=1)
        hash_nope = self.hash_encoder_nope.compute_hash(nope)
        hash_rope = self.hash_encoder_rope.compute_hash(rope)
        return hash_nope.view(torch.bfloat16), hash_rope.view(torch.bfloat16)

    # ------------------------------------------------------------------
    # cache_k_hash  (called by attention_begin)
    # ------------------------------------------------------------------

    def cache_k_hash(
        self,
        nope: torch.Tensor,
        rope: torch.Tensor,
        k_hash: torch.Tensor,
        attn_metadata,
        forward_context: ForwardContext,
        layer_name: str,
    ):
        k_c_normed_hash, k_pe_hash = self.hash_code(nope=nope, rope=rope)
        ops.concat_and_cache_mla(
            k_c_normed_hash,
            k_pe_hash.squeeze(1),
            k_hash,
            attn_metadata.slot_mapping.flatten(),
            kv_cache_dtype="auto",
            scale=self._k_scale,
        )
        if self.has_pc_hit:
            attn = forward_context.no_compile_layers[layer_name]
            kv_cache = attn.kv_cache[forward_context.virtual_engine]
            k_cache = kv_cache[0][self.prefix_block_ids]
            k_c_normed, k_pe = torch.split(k_cache, [512, 64], dim=-1)
            k_c_normed = k_c_normed.reshape(-1, k_c_normed.shape[2])
            k_pe = k_pe.reshape(-1, k_pe.shape[2])
            k_c_normed_hash, k_pe_hash = self.hash_code(nope=k_c_normed, rope=k_pe)
            ops.concat_and_cache_mla(
                k_c_normed_hash,
                k_pe_hash,
                k_hash,
                self.prefix_slot_mapping.flatten(),
                kv_cache_dtype="auto",
                scale=self._k_scale,
            )

    # ------------------------------------------------------------------
    # update_decode_topk  (called by attention_begin for decode phase)
    # ------------------------------------------------------------------

    def update_decode_topk(
        self,
        is_rollback_layer: bool,
        is_skip_hash_layer: bool,
        attn_metadata,
        decode_ql_nope: torch.Tensor,
        decode_q_pe: torch.Tensor,
        k_hash: torch.Tensor,
    ):
        if not is_rollback_layer:
            if is_skip_hash_layer:
                assert attn_metadata.decode.topk_block_table is not None
                block_table = attn_metadata.decode.topk_block_table
            else:
                q_nope_hash, q_rope_hash = self.hash_code(
                    nope=decode_ql_nope,
                    rope=decode_q_pe,
                    reduction_head_num=self.hash_reduction_head_num,
                )
                q_hash = torch.cat([q_nope_hash, q_rope_hash], dim=-1)
                block_table = cuda_hamming_topk(
                    q_hash.unsqueeze(1),
                    k_hash.unsqueeze(2),
                    attn_metadata.decode.block_table,
                    attn_metadata.decode.seq_lens,
                    topk_token=self.hash_topk_tokens,
                    sink_token=self.block_size,
                    recent_token=self.block_size * 4,
                    is_mla=True,
                )
                attn_metadata.decode.topk_block_table = block_table

            seq_lens = attn_metadata.decode.topk_seq_lens
            tile_scheduler_metadata = attn_metadata.decode.topk_tile_scheduler_metadata
            num_splits = attn_metadata.decode.topk_num_splits

            self.ori_block_table_decode = attn_metadata.decode.block_table
            self.ori_seq_lens_decode = attn_metadata.decode.seq_lens
            self.origin_tile_scheduler_metadata = (
                attn_metadata.decode.tile_scheduler_metadata
            )
            self.origin_num_splits = attn_metadata.decode.num_splits

            attn_metadata.decode.block_table = block_table
            attn_metadata.decode.seq_lens = seq_lens
            attn_metadata.decode.tile_scheduler_metadata = tile_scheduler_metadata
            attn_metadata.decode.num_splits = num_splits

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
            self.cache_k_hash(
                nope=key,
                rope=value,
                k_hash=k_hash,
                attn_metadata=attn_metadata,
                forward_context=forward_context,
                layer_name=layer_name,
            )

        if not self.gsa_enabled:
            return query, key, value, output

        if phase == "decode":
            self.update_decode_topk(
                is_rollback_layer,
                is_skip_hash_layer,
                attn_metadata,
                decode_ql_nope,
                decode_q_pe,
                k_hash,
            )

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
        if phase == "decode":
            is_rollback_layer, _ = self.get_layer_state(layer_name)
            if not is_rollback_layer:
                attn_metadata = self.get_layer_attn_metadata(forward_context, layer_name)
                attn_metadata.decode.block_table = self.ori_block_table_decode
                attn_metadata.decode.seq_lens = self.ori_seq_lens_decode
                attn_metadata.decode.tile_scheduler_metadata = (
                    self.origin_tile_scheduler_metadata
                )
                attn_metadata.decode.num_splits = self.origin_num_splits

    # ------------------------------------------------------------------
    # initialize_kv_hash_cache_tensors
    # ------------------------------------------------------------------

    def initialize_kv_hash_cache_tensors(self, kv_caches, device):
        dtype = torch.bfloat16
        for layer_name, kv_cache in kv_caches.items():
            is_rollback_layer, is_skip_hash_layer = self.get_layer_state(layer_name)
            if not is_rollback_layer and not is_skip_hash_layer:
                khash_cache_shape = list(kv_cache.shape)
                khash_cache_shape[-1] //= dtype.itemsize * 8
                khash_cache = torch.zeros(khash_cache_shape, dtype=dtype, device=device)
            else:
                khash_cache = None
            kv_caches[layer_name] = (kv_cache, khash_cache)

    # ------------------------------------------------------------------
    # get_block_table_row / get_seq_lens
    # ------------------------------------------------------------------

    def get_block_table_row(self, attn_metadata, req_row_id, prefill_row_id):
        attn_metadata_prefill = getattr(attn_metadata, "prefill", None)
        return attn_metadata_prefill.block_table[prefill_row_id]

    def get_seq_lens(self, attn_metadata):
        attn_metadata_decode = getattr(attn_metadata, "decode", None)
        if attn_metadata_decode is not None:
            return getattr(attn_metadata_decode, "seq_lens", None)

        attn_metadata_prefill = getattr(attn_metadata, "prefill", None)
        if attn_metadata_prefill is None:
            return None

        chunked = getattr(attn_metadata_prefill, "chunked_context", None)
        if chunked is not None:
            return getattr(chunked, "seq_lens", None)

        # first-time prefill (non-chunked)
        query_start_loc_prefill = getattr(attn_metadata_prefill, "query_start_loc", None)
        if query_start_loc_prefill is not None:
            return query_start_loc_prefill[1:] - query_start_loc_prefill[:-1]

        return getattr(attn_metadata_prefill, "seq_lens", None)

    # ------------------------------------------------------------------
    # build_decode_hash (CUDA MLA only)
    # ------------------------------------------------------------------

    def build_decode_hash(self, seq_lens):
        topk_seq_lens = update_seq_lens(
            seq_lens,
            topk_token=self.hash_topk_tokens,
            block_size=self.block_size,
        )
        topk_tile_scheduler_metadata, topk_num_splits = get_mla_metadata(
            topk_seq_lens,
            self.num_q_heads,
            1,
        )
        return topk_seq_lens, topk_tile_scheduler_metadata, topk_num_splits

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

        self._run_build_sparse_meta_loop(
            scheduler_output, requests, input_batch, attn_metadata
        )
        # MLA: decode sparse metadata is handled entirely in attention_begin;
        # no further work needed here.

    # ------------------------------------------------------------------
    # execute_begin / execute_finished
    # ------------------------------------------------------------------

    def execute_begin(self, scheduler_output: SchedulerOutput):
        pass  # no per-step reset needed on CUDA MLA

    def execute_finished(self, logits_indices: torch.Tensor):
        self.gsa_enabled = False
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
