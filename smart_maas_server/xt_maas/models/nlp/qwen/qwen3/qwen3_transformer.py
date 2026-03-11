from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Qwen3_TransformerModel:
    THINK_START = 151668  # <|think|>
    THINK_END   = 151669  # </think|>

    def __init__(self,
                 model_name: str = "Qwen/Qwen3-8B",
                 torch_dtype="auto",
                 device_map="auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map
        )

    def generate(self,
                prompt: str,
                max_new_tokens: int = 32768,
                temperature: float = 0.7,
                top_p: float = 0.8,
                enable_thinking: bool = False,
                **gen_kwargs) -> Tuple[str, str]:

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )
        output_ids = generated_ids[0][model_inputs.input_ids.shape[1]:].tolist()

        # 解析think块
        try:
            end_idx = len(output_ids) - output_ids[::-1].index(self.THINK_END)
        except ValueError:
            end_idx = 0

        thinking = self.tokenizer.decode(output_ids[:end_idx], skip_special_tokens=True).strip()
        content = self.tokenizer.decode(output_ids[end_idx:], skip_special_tokens=True).strip()
        result = {}
        result["thinking"] = thinking
        result["gen_tokens"] = len(output_ids)
        result["gen_text"] = content

        return result