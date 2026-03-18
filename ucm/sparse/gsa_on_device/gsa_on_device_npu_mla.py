"""
NPU + MLA (DeepSeek Multi-Latent Attention) implementation of GSAOnDevice.
Inherits from GSAOnDeviceCudaMLA and overrides NPU-specific methods.
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
from ucm.sparse.gsa_on_device.gsa_on_device_cuda_mla import GSAOnDeviceCudaMLA
from ucm.sparse.gsa_on_device.hamming_topk import update_seq_lens

logger = init_logger(__name__)

ReqType = Union[str, int]


class GSAOnDeviceNpuMLA(GSAOnDeviceCudaMLA):
    """
    GSAOnDevice for NPU + MLA (DeepSeek) models.
    Overrides all CUDA-specific methods from GSAOnDeviceCudaMLA with NPU equivalents.
    """

    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        # Call GSAOnDeviceBase.__init__ directly, bypassing CUDA-specific MLA init,
        # then run the NPU+MLA-specific initialization.
        from ucm.sparse.gsa_on_device.gsa_on_device_base import GSAOnDeviceBase
        GSAOnDeviceBase.__init__(self, vllm_config, role, is_cuda=False)

        logger.info("GSAOnDevice initialized with MLA model config (NPU)")

        if role != UcmSparseRole.WORKER:
            return

        from ucm.sparse.gsa_on_device.hash_encoder import HashEncoder
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

        # NPU MLA specific
        self.khash_zeros_full = None
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
    # cache_k_hash  (NPU MLA version)
    # ------------------------------------------------------------------

    def cache_k_hash(
        self,
        nope: torch.Tensor,
        rope: torch.Tensor,
        k_hash,
        attn_metadata,
        forward_context: ForwardContext,
        layer_name: str,
    ):
        khash_nope_cache, khash_rope_cache = k_hash

        khash_nope = self.hash_encoder_nope.compute_hash(nope)
        khash_rope = self.hash_encoder_rope.compute_hash(rope)
        khash_nope_new = (
            khash_nope.transpose(0, 1).reshape(-1, khash_nope.shape[-1]).contiguous()
        )
        khash_rope_new = (
            khash_rope.transpose(0, 1).reshape(-1, khash_rope.shape[-1]).contiguous()
        )

        if (
            self.khash_zeros_full is None
            or self.khash_zeros_full.numel() < khash_rope_new.numel()
        ):
            self.khash_zeros_full = torch.zeros_like(khash_rope_new)

        khash_zeros = self.khash_zeros_full[: khash_rope_new.shape[0]]
        khash_rope_new_pad = torch.cat((khash_rope_new, khash_zeros), dim=-1).contiguous()

        # Determine slot_mapping and num_tokens_device from phase context
        # (set by attention_begin before calling cache_k_hash)
        torch.ops._C_ucm.npu_reshape_and_cache_bnsd(
            khash_nope_new,
            khash_nope_cache,
            self._slot_mapping,
            self._num_tokens_device,
            khash_nope_cache,
        )
        torch.ops._C_ucm.npu_reshape_and_cache_bnsd(
            khash_rope_new_pad,
            khash_rope_cache,
            self._slot_mapping,
            self._num_tokens_device,
            khash_rope_cache,
        )

        if self.has_pc_hit:
            attn = forward_context.no_compile_layers[layer_name]
            (nope_cache, rope_cache), khash_cache = attn.kv_cache[
                forward_context.virtual_engine
            ]
            k_nope_cache = (
                nope_cache[self.prefix_block_ids]
                .reshape(-1, nope_cache.shape[2], nope_cache.shape[3])
                .unsqueeze(1)
            )
            k_rope_cache = (
                rope_cache[self.prefix_block_ids]
                .reshape(-1, rope_cache.shape[2], rope_cache.shape[3])
                .unsqueeze(1)
            )
            khash_nope = self.hash_encoder_nope.compute_hash(k_nope_cache)
            khash_rope = self.hash_encoder_rope.compute_hash(k_rope_cache)
            khash_nope_new = (
                khash_nope.transpose(0, 1)
                .reshape(-1, khash_nope.shape[-1])
                .contiguous()
            )
            khash_rope_new = (
                khash_rope.transpose(0, 1)
                .reshape(-1, khash_rope.shape[-1])
                .contiguous()
            )
            if (
                self.khash_zeros_full is None
                or self.khash_zeros_full.numel() < khash_rope_new.numel()
            ):
                self.khash_zeros_full = torch.zeros_like(khash_rope_new)
            khash_zeros = self.khash_zeros_full[: khash_rope_new.shape[0]]
            khash_rope_new_pad = torch.cat(
                (khash_rope_new, khash_zeros), dim=-1
            ).contiguous()
            torch.ops._C_ucm.npu_reshape_and_cache_bnsd(
                khash_nope_new,
                khash_nope_cache,
                self.prefix_slot_mapping.flatten(),
                self.prefix_token_len,
                khash_nope_cache,
            )
            torch.ops._C_ucm.npu_reshape_and_cache_bnsd(
                khash_rope_new_pad,
                khash_rope_cache,
                self.prefix_slot_mapping.flatten(),
                self.prefix_token_len,
                khash_rope_cache,
            )

    # ------------------------------------------------------------------
    # update_decode_topk  (NPU MLA version)
    # ------------------------------------------------------------------

    def update_decode_topk(
        self,
        is_rollback_layer: bool,
        is_skip_hash_layer: bool,
        attn_metadata,
        decode_ql_nope: torch.Tensor,
        decode_q_pe: torch.Tensor,
        k_hash,
    ):
        if not self.is_tensor_computed:
            topk_device = cdiv(
                attn_metadata.decode.seq_lens_device, self.block_size
            ).to(dtype=torch.int32)
            self.topk_device = torch.clamp(
                topk_device, min=1, max=self.hash_topk_tokens // self.block_size
            )
            self.is_tensor_computed = True

        if not is_rollback_layer:
            if is_skip_hash_layer:
                attn_metadata.decode.block_table = self.topk_block_table
            else:
                khash_nope_cache, khash_rope_cache = k_hash
                batch_size = attn_metadata.num_decodes

                qhash_nope = self.hash_encoder_nope.compute_hash(decode_ql_nope)
                qhash_rope = self.hash_encoder_rope.compute_hash(decode_q_pe)
                qhash_zeros = torch.zeros_like(qhash_rope)
                qhash_pad = (
                    torch.cat((qhash_nope, qhash_rope, qhash_zeros), dim=-1)
                    .unsqueeze(2)
                    .contiguous()
                )

                new_block_table = torch.ops._C_ucm.npu_hamming_dist_top_k(
                    qhash_pad,
                    khash_nope_cache,
                    khash_rope_cache,
                    self.topk_device,
                    attn_metadata.decode.seq_lens_device,
                    self.chunk_sizes_for_hamming_full[:batch_size],
                    attn_metadata.decode.max_seq_lens,
                    self.hamming_keep_chunks_head,
                    self.hamming_keep_chunks_tail,
                    0,  # not support offload
                    self.ori_block_table_decode[:batch_size],
                    None,
                    self.hamming_output[:batch_size],
                )
                attn_metadata.decode.block_table = new_block_table[:, 0, :]

            self.topk_block_table = attn_metadata.decode.block_table
            attn_metadata.decode.seq_lens = self.new_seq_lens
            attn_metadata.decode.seq_lens_list = self.new_seq_lens_list

    # ------------------------------------------------------------------
    # attention_begin: NPU MLA needs to set slot_mapping/num_tokens_device
    # before calling cache_k_hash
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
        k_hash=None,
        decode_ql_nope: Optional[torch.Tensor] = None,
        decode_q_pe: Optional[torch.Tensor] = None,
    ):
        attn_metadata = self.get_layer_attn_metadata(forward_context, layer_name)
        is_rollback_layer, is_skip_hash_layer = self.get_layer_state(layer_name)

        if not is_rollback_layer and not is_skip_hash_layer:
            if phase == "decode":
                self._slot_mapping = attn_metadata.slot_mapping[
                    : attn_metadata.num_decode_tokens
                ]
                self._num_tokens_device = attn_metadata.num_decode_tokens_device
            else:
                self._slot_mapping = attn_metadata.slot_mapping[
                    attn_metadata.num_decode_tokens: attn_metadata.num_actual_tokens
                ]
                self._num_tokens_device = attn_metadata.num_prefill_tokens_device
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

    # ------------------------------------------------------------------
    # attention_finished: NPU MLA restores seq_lens_list instead of
    # tile_scheduler_metadata / num_splits
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
        if phase == "decode":
            is_rollback_layer, _ = self.get_layer_state(layer_name)
            if not is_rollback_layer:
                attn_metadata = self.get_layer_attn_metadata(forward_context, layer_name)
                attn_metadata.decode.block_table = self.ori_block_table_decode
                attn_metadata.decode.seq_lens = self.ori_seq_lens_decode
                attn_metadata.decode.seq_lens_list = self.ori_seq_lens_list

    # ------------------------------------------------------------------
    # initialize_kv_hash_cache_tensors (NPU MLA version)
    # ------------------------------------------------------------------

    def initialize_kv_hash_cache_tensors(self, kv_caches, device):
        for layer_name, kv_cache in kv_caches.items():
            is_rollback_layer, is_skip_hash_layer = self.get_layer_state(layer_name)
            kv_cache_nope_shape = kv_cache[0].shape
            kv_cache_rope_shape = kv_cache[1].shape
            khash_nope_shape = (
                kv_cache_nope_shape[0],
                kv_cache_nope_shape[2],
                kv_cache_nope_shape[1],
                self.hash_encoder_nope.hash_bits // 8,
            )
            khash_rope_shape = (
                kv_cache_rope_shape[0],
                kv_cache_rope_shape[2],
                kv_cache_rope_shape[1],
                self.hash_encoder_rope.hash_bits // 8 * 2,  # *2 for NPU padding
            )
            if not is_rollback_layer and not is_skip_hash_layer:
                khash_nope_cache = torch.empty(
                    khash_nope_shape, dtype=torch.uint8, device=device
                )
                khash_rope_cache = torch.empty(
                    khash_rope_shape, dtype=torch.uint8, device=device
                )
                khash_cache = (khash_nope_cache, khash_rope_cache)
            else:
                khash_cache = None
            kv_caches[layer_name] = (kv_cache, khash_cache)

    # ------------------------------------------------------------------
    # get_block_table_row: MLA uses prefill sub-metadata
    # ------------------------------------------------------------------

    def get_block_table_row(self, attn_metadata, req_row_id, prefill_row_id):
        attn_metadata_prefill = getattr(attn_metadata, "prefill", None)
        return attn_metadata_prefill.block_table[prefill_row_id]

    # ------------------------------------------------------------------
    # get_seq_lens: NPU MLA uses seq_lens on prefill sub-metadata
    # ------------------------------------------------------------------

    def get_seq_lens(self, attn_metadata):
        attn_metadata_decode = getattr(attn_metadata, "decode", None)
        if attn_metadata_decode is not None:
            return getattr(attn_metadata_decode, "seq_lens", None)

        attn_metadata_prefill = getattr(attn_metadata, "prefill", None)
        if attn_metadata_prefill is None:
            return None
        return getattr(attn_metadata_prefill, "seq_lens", None)

    # ------------------------------------------------------------------
    # build_decode_attention_meta_npu (called externally by integration layer)
    # ------------------------------------------------------------------

    def build_decode_attention_meta_npu(self, query_lens, seq_lens, block_table):
        self.ori_seq_lens_decode = seq_lens.clone()
        self.ori_block_table_decode = block_table.clone()

        self.new_seq_lens = update_seq_lens(
            seq_lens,
            topk_token=self.hash_topk_tokens,
            block_size=self.block_size,
        )
        self.ori_seq_lens_list = seq_lens.tolist()
        self.new_seq_lens_list = self.new_seq_lens.tolist()

    # ------------------------------------------------------------------
    # execute_begin: reset is_tensor_computed each step (NPU only)
    # ------------------------------------------------------------------

    def execute_begin(self, scheduler_output: SchedulerOutput):
        self.is_tensor_computed = False

    # ------------------------------------------------------------------
    # execute_finished: no has_decode / decode_only on MLA
    # ------------------------------------------------------------------

    def execute_finished(self, logits_indices: torch.Tensor):
        self.gsa_enabled = False
        self.has_pc_hit = False
        return logits_indices
