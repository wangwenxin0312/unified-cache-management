import contextlib
import os
import time
from dataclasses import asdict
import json
from transformers import AutoTokenizer

# Third Party
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from ucm.logger import init_logger

logger = init_logger(__name__)


@contextlib.contextmanager
def build_llm_with_uc(module_path: str, name: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=name,
        kv_connector_module_path=module_path,
        kv_role="kv_both",
        kv_connector_extra_config={"UCM_CONFIG_FILE": "./ucm_config_example.yaml",
                                   "use_layerwise":False,
        },
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=30000,
        gpu_memory_utilization=0.85,
        max_num_batched_tokens=30000,
        block_size=128,
        enforce_eager=True,
        trust_remote_code=True,
        tensor_parallel_size=4,
        enable_prefix_caching=False,
        mamba_cache_mode="align",
        disable_hybrid_kv_cache_manager=False,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        logger.info("LLM engine is exiting.")


def print_output(
    llm: LLM,
    prompt: list[str],
    sampling_params: SamplingParams,
    req_str: str,
):
    start = time.time()
    outputs = llm.generate(prompt, sampling_params)
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")
    print("-" * 50)


def main():
    module_path = "ucm.integration.vllm.ucm_connector"
    name = "UCMConnector"
    model = os.getenv("MODEL_PATH", "/home/models/Qwen3-Next-80B-A3B-Instruct/")

    tokenizer = AutoTokenizer.from_pretrained(model, use_chat_template=True)
    
    def get_prompt(prompt):
        messages = [
            {
                "role": "system",
                "content": "先读问题，再根据下面的文章内容回答问题，不要进行分析，不要重复问题，用简短的语句给出答案。\n\n例如：“全国美国文学研究会的第十八届年会在哪所大学举办的？”\n回答应该为：“xx大学”。\n\n",
            },
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=True,
        )
    
    with build_llm_with_uc(module_path, name, model) as llm:
        prompts = []
        batch_size = 20
        with open("/home/data/Longbench/data/multifieldqa_zh.jsonl", "r") as f:
            lines = f.readlines()
        for i in range(batch_size):
            line = lines[i]
            data = json.loads(line)
            prompt = f"""阅读以下文字并用中文简短回答：\n\n{data["context"]}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{data["input"]}\n回答："""
            prompts.append(get_prompt(prompt))

        sampling_params = SamplingParams(
            temperature=0, top_p=0.95, max_tokens=256, ignore_eos=False
        )

        print_output(llm, prompts, sampling_params, "first")
        print_output(llm, prompts, sampling_params, "second")


if __name__ == "__main__":
    main()
