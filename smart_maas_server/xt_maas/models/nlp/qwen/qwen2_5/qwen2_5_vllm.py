import os
from vllm import LLM, SamplingParams

# 允许超长上下文(显卡条件允许的情况)
os.environ["VLLM_ATTENTION_BACKEND"] = "DUAL_CHUNK_FLASH_ATTN"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"


class Qwen2_5_VllmModel:
    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-7B-Instruct-1M",
        tensor_parallel_size: int = 1,
        max_model_len: int = 45000,
        gpu_memory_utilization: float = 0.95,
        max_num_batched_tokens: int = 8192,
        **vllm_kwargs,
    ):

        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=max_num_batched_tokens,
            enforce_eager=True,
            enable_chunked_prefill=True,
            **vllm_kwargs,
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.05,
        max_new_tokens: int = 256,
        **sampling_kwargs,
    ):
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_new_tokens,
            detokenize=True,
            **sampling_kwargs,
        )

        outs = self.llm.generate([prompt], sampling_params)

        result = {}
        result["thinking"] = ""
        result["gen_tokens"] = len(outs[0].outputs[0].token_ids)
        result["gen_text"] = outs[0].outputs[0].text

        return result