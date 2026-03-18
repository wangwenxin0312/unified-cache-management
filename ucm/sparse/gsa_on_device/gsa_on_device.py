"""
GSAOnDevice factory.

Dispatches to the correct implementation class based on model type (GQA / MLA)
and hardware platform (CUDA / NPU):

    ┌───────────┬──────────────────────────┬──────────────────────────┐
    │           │         CUDA             │         NPU              │
    ├───────────┼──────────────────────────┼──────────────────────────┤
    │  GQA      │ GSAOnDeviceCudaGQA       │ GSAOnDeviceNpuGQA        │
    │  MLA      │ GSAOnDeviceCudaMLA       │ GSAOnDeviceNpuMLA        │
    └───────────┴──────────────────────────┴──────────────────────────┘

Public re-exports keep existing import paths working:

    from ucm.sparse.gsa_on_device.gsa_on_device import GSAOnDevice
    from ucm.sparse.gsa_on_device.gsa_on_device import gsa_on_device_config_path_for_model
"""
from vllm.config import VllmConfig

from ucm.sparse.base import UcmSparseRole
from ucm.sparse.gsa_on_device.gsa_on_device_base import (
    GSAOnDeviceBase,
    gsa_on_device_config_path_for_model,
)
from ucm.sparse.gsa_on_device.gsa_on_device_cuda_gqa import GSAOnDeviceCudaGQA
from ucm.sparse.gsa_on_device.gsa_on_device_cuda_mla import GSAOnDeviceCudaMLA

__all__ = [
    "GSAOnDevice",
    "GSAOnDeviceBase",
    "GSAOnDeviceCudaGQA",
    "GSAOnDeviceCudaMLA",
    "gsa_on_device_config_path_for_model",
]


def GSAOnDevice(vllm_config: VllmConfig, role: UcmSparseRole):
    """
    Factory function that returns the appropriate GSAOnDevice instance.

    Selection logic:
      - device:  cuda  → CUDA variants
      - device:  npu   → NPU variants
      - model:   MLA   → MLA variants  (deepseek_mla flag)
      - model:   GQA   → GQA variants  (all others)
    """
    device_type = vllm_config.device_config.device_type
    is_mla = vllm_config.model_config.is_deepseek_mla

    if device_type == "cuda":
        if is_mla:
            return GSAOnDeviceCudaMLA(vllm_config, role)
        else:
            return GSAOnDeviceCudaGQA(vllm_config, role)

    elif device_type == "npu":
        # Lazy import to avoid importing torch_npu on CUDA environments
        if is_mla:
            from ucm.sparse.gsa_on_device.gsa_on_device_npu_mla import (
                GSAOnDeviceNpuMLA,
            )
            return GSAOnDeviceNpuMLA(vllm_config, role)
        else:
            from ucm.sparse.gsa_on_device.gsa_on_device_npu_gqa import (
                GSAOnDeviceNpuGQA,
            )
            return GSAOnDeviceNpuGQA(vllm_config, role)

    else:
        raise ValueError(
            f"[GSAOnDevice] Unsupported device type: {device_type}. "
            "Expected 'cuda' or 'npu'."
        )
