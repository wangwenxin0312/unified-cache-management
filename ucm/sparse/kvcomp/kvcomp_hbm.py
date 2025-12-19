from typing import Any, Dict, List, Optional, Union

import torch
from vllm.config import VllmConfig
from vllm import _custom_ops as ops
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.forward_context import ForwardContext
import vllm.envs as envs


from ucm.logger import init_logger
from ucm.sparse.base import (
    INVALID_SLOT,
    UcmSparseBase,
    UcmSparseRole,
)

from ucm.sparse.kvcomp.hash_code import triton_hash_code
from ucm.sparse.kvcomp.hamming_topk import cuda_hamming_topk, fake_hamming_topk
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)

ReqType = Union[str, int]
class KvComp_NonOffload(UcmSparseBase):
    # handle batch
    def __init__(self, vllm_config: VllmConfig, role: UcmSparseRole):
        super().__init__(vllm_config, role)
        # self.rank = vllm_config.parallel_config.rank
        
        self.om_hash_nope = None  # [512, 512]
        self.om_hash_rope = None  # [64, 64]
        self.pack_weight = None
        self.kv_lora_rank = 512
        self.qk_rope_head_dim = 64
        self.origin_attn_metadata = None

        if self.om_hash_nope is None:
            def gen_matrix(dim):
                low_dim = dim
                A = torch.normal(0, 2, (dim, low_dim), dtype=torch.float32).to(torch.cuda.current_device())
                Q, R = torch.linalg.qr(A)
                d = torch.sign(torch.diag(R))
                Q*=d
                return Q.to(torch.bfloat16)
            self.om_hash_nope = gen_matrix(self.kv_lora_rank)
            self.om_hash_rope = gen_matrix(self.qk_rope_head_dim)
            self.pack_weight = torch.pow(
                2, torch.arange(8, device=torch.cuda.current_device()).to(torch.uint8))

    def hash_code(self, nope, rope, reduction_head_num=1):
        if reduction_head_num > 1:
            nope = nope.view(nope.shape[0], reduction_head_num,
                             nope.shape[1] // reduction_head_num, nope.shape[2]).mean(dim=1)
            rope = rope.view(rope.shape[0], reduction_head_num,
                             rope.shape[1] // reduction_head_num, rope.shape[2]).mean(dim=1)

        hash_nope = triton_hash_code(nope, self.om_hash_nope, self.pack_weight)
        hash_rope = triton_hash_code(rope, self.om_hash_rope, self.pack_weight)
        return hash_nope.view(torch.bfloat16), hash_rope.view(torch.bfloat16)
    
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
        k_scale: Optional[torch.Tensor] = None,
        decode_ql_nope: Optional[torch.Tensor] = None,
        decode_q_pe: Optional[torch.Tensor] = None,
    ):
        attn_metadata = forward_context.attn_metadata # todo:dict
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[layer_name]

        layer_id = int(layer_name.split('.')[2])
        # TODO: Should mark MTP layer as rollback layer
        is_rollback_layer = layer_id in envs.VLLM_HASH_ATTENTION_ROLLBACK_LAYERS
        is_skip_hash_layer = layer_id < len(envs.VLLM_HASH_ATTENTION_SKIP_LAYERS) and \
            envs.VLLM_HASH_ATTENTION_SKIP_LAYERS[layer_id]
        
        if envs.VLLM_HASH_ATTENTION and not is_rollback_layer and not is_skip_hash_layer:
                k_c_normed_hash, k_pe_hash = self.hash_code(key, value)
                ops.concat_and_cache_mla(
                    k_c_normed_hash,
                    k_pe_hash.squeeze(1),
                    k_hash,
                    attn_metadata.slot_mapping.flatten(),
                    kv_cache_dtype='auto',
                    scale=k_scale,
                )
        if phase == "decode":
            if envs.VLLM_HASH_ATTENTION and not is_rollback_layer:
                if is_skip_hash_layer:
                    assert attn_metadata.decode.topk_block_table is not None
                    block_table = attn_metadata.decode.topk_block_table
                else:
                    q_hash = torch.cat(self.hash_code(decode_ql_nope, decode_q_pe,
                                                      reduction_head_num=envs.VLLM_HASH_ATTENTION_REDUCTION_HEAD_NUM), dim=-1)
                    topk_token = envs.VLLM_HASH_ATTENTION_TOPK
                    block_table = cuda_hamming_topk(
                        q_hash.unsqueeze(1),
                        k_hash.unsqueeze(1),
                        attn_metadata.decode.block_table,
                        attn_metadata.decode.seq_lens,
                        topk_token=topk_token,
                        sink_token=64,
                        recent_token=512,
                    )
                    attn_metadata.decode.topk_block_table = block_table

                seq_lens = attn_metadata.decode.topk_seq_lens
                tile_scheduler_metadata = attn_metadata.decode.topk_tile_scheduler_metadata
                num_splits = attn_metadata.decode.topk_num_splits

                self.origin_attn_metadata = attn_metadata.decode

                attn_metadata.decode.block_table = block_table
                attn_metadata.decode.seq_lens = seq_lens
                attn_metadata.decode.tile_scheduler_metadata = tile_scheduler_metadata
                attn_metadata.decode.num_splits = num_splits
        
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
        if phase == "decode":
            layer_id = int(layer_name.split('.')[2])
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[layer_name]
            # TODO: Should mark MTP layer as rollback layer
            is_rollback_layer = layer_id in envs.VLLM_HASH_ATTENTION_ROLLBACK_LAYERS
            if envs.VLLM_HASH_ATTENTION and not is_rollback_layer:
                attn_metadata.decode.block_table = self.origin_attn_metadata.block_table
                attn_metadata.decode.seq_lens = self.origin_attn_metadata.seq_lens
                attn_metadata.decode.tile_scheduler_metadata = self.origin_attn_metadata.tile_scheduler_metadata
                attn_metadata.decode.num_splits = self.origin_attn_metadata.num_splits

    def request_begin(self, request_id: ReqType, prompt_token_ids: List[int]):
        pass

    def request_finished_in_scheduler(self, request_id: Union[int, str]):
        """
        This is called inside "Scheduler->finish_requests" function.
        Generate the metadata required by UcmSparse instance at worker-side.
        """
        pass

    def estimate_num_slots_sparsed(self, request: Request) -> int:
        return INVALID_SLOT

    def initialize_kv_hash_cache_tensors(self, kv_caches, device):
        if envs.VLLM_HASH_ATTENTION:
            dtype = torch.bfloat16
            for layer_name, kv_cache in kv_caches.items():
                khash_cache_shape = list(kv_cache.shape)
                khash_cache_shape[-1] //= dtype.itemsize * 8
                khash_cache = torch.zeros(khash_cache_shape,
                                            dtype=dtype,
                                            device=device)
                kv_caches[layer_name] = (kv_cache, khash_cache)