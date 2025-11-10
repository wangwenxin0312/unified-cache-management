import contextlib
import os
import time
from dataclasses import asdict

# Third Party
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from ucm.logger import init_logger
import json
from transformers import AutoTokenizer
import shutil
logger = init_logger(__name__)
data_dir = "/home/externals/wangwenxin21/unified-cache-management/datasets/"

def setup_environment_variables():
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["PYTHONHASHSEED"] = "123456"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # os.system("export VLLM_LOGGING_LEVEL=DEBUG")
    os.system("rm -rf ./ucm/data/*")


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                data.append(json.loads(line))
    return data

@contextlib.contextmanager
def build_llm_with_uc(module_path: str, name: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=name,
        kv_connector_module_path=module_path,
        kv_role="kv_both",
        kv_connector_extra_config={
            "ucm_connector_name": "UcmNfsStore",
            "ucm_connector_config": {
                "storage_backends": data_dir,
            },
            "ucm_sparse_config": {
                "GSA": {
                }
            }, 
        },
    )

    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=70000,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=30000,
        block_size=128,
        enforce_eager=True,
        distributed_executor_backend="mp",
        tensor_parallel_size=1,
        trust_remote_code=True
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
    return generated_text


def main():
    test_num = 200
    test_max_len = 69000
    module_path = "ucm.integration.vllm.uc_connector"
    name = "UnifiedCacheConnectorV1"
    # model = os.getenv("MODEL_PATH", "/home/models/DeepSeek-R1-Distill-Qwen-32B/")
    model = os.getenv("MODEL_PATH", "/home/models/DeepSeek-V2-Lite-Chat/")
    # ans_out_path = "dqa_zh_ans_gsa_1.json"
    setup_environment_variables()
    tokenizer = AutoTokenizer.from_pretrained("/home/models/DeepSeek-V2-Lite-Chat/", use_chat_template = True)
    sparse_method = "gsa-deepseek-v2-lit-chat-pc"
    
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
        # multifieldqa_zh
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1024, ignore_eos=False)
        jsonl_data = read_jsonl("/home/externals/wangwenxin21/datasets/LongBench/data/multifieldqa_zh.jsonl")
        ans_out_path = f"DeepSeek-R1-Distill-Qwen-32B_{sparse_method}_multifieldqa_zh_new_token"
        for i, item in enumerate(jsonl_data):
            shutil.rmtree("/home/externals/wangwenxin21/unified-cache-management/datasets/data")
            os.makedirs("/home/externals/wangwenxin21/unified-cache-management/datasets/data")
            answer = item["answers"]
            length = len(item["context"])
            context_token = tokenizer.encode(item["context"])
            # if len(context_token) > test_max_len: 
            #     continue
            print(f"start doc {i} infer")
            # 改进prompt设计，更明确的指示
            prompt = f"""阅读以下文字并用中文简短回答：\n\n{item["context"]}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{item["input"]}\n回答："""
            prompt = get_prompt(prompt)
            start = time.perf_counter()
            generated_text = print_output(llm, prompt, sampling_params, str(i))
            print(f"E2E_time:{time.perf_counter() - start}")
            with open(ans_out_path, "a", encoding="utf-8") as f:
                json.dump({"pred": generated_text, "answers": answer, "length": length, "E2E_time": time.perf_counter() - start, "context_token": len(context_token)}, f, ensure_ascii=False)
                f.write('\n')
                f.close()
                
if __name__ == "__main__":
    main()