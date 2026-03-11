import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Qwen2_5_TransformerModel:
    def __init__(self,
                 model_path: str = "Qwen/Qwen2.5-7B-Instruct-1M",
                 device_map: str = "auto", 
                 torch_dtype=torch.bfloat16,
                 trust_remote_code=True):

        self.model  = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code)

        self.model.eval()

    def generate(self,
                 prompt: str,
                 max_new_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.8,
                 top_k: int = 20,
                 repetition_penalty: float = 1.05,
                 **kwargs) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]       
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # 截断到模型最大长度
        max_in = self.model.config.max_position_embeddings - max_new_tokens
        if model_inputs.input_ids.shape[1] > max_in:
            model_inputs["input_ids"] = model_inputs.input_ids[:, -max_in:]
            model_inputs["attention_mask"] = model_inputs.attention_mask[:, -max_in:]
        
        gen = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs)
        
        new_ids = gen[0][model_inputs.input_ids.shape[1]:]
        out = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        result = {}
        result["thinking"] = ""
        result["gen_tokens"] = len(new_ids)
        result["gen_text"] = out

        return result