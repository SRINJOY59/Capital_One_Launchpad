import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from peft import PeftModel
import os
import warnings
warnings.filterwarnings("ignore")

class HFModel:
    def __init__(self, base_model_dir: str, adapter_dir: str, device: str = "auto"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model_dir = os.path.abspath(base_model_dir)
        self.adapter_dir = os.path.abspath(adapter_dir)
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        # Load base model and tokenizer
        if os.path.exists(self.base_model_dir):
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_dir, trust_remote_code=True, local_files_only=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=None,
                trust_remote_code=True,
                local_files_only=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen1.5-1.8B-Chat",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=None,
                trust_remote_code=True
            )

        # Load PEFT adapter
        model = PeftModel.from_pretrained(base_model, self.adapter_dir)
        model = model.to(self.device)
        return model, tokenizer

    def infer(self, prompt: str, max_new_tokens: int = 200):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    base_model_dir = "./models/Qwen1.5-Base"
    adapter_dir = "./models/Qwen_1.5_Finetuned"
    hf_model = HFModel(base_model_dir, adapter_dir)
    prompt = "Explain how AI can help farmers increase crop yield."
    output = hf_model.infer(prompt)
    print("\n=== Model Output ===\n")
    print(output)