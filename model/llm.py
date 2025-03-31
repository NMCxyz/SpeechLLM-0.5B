from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    Qwen2ForCausalLM,
    Qwen2Tokenizer
)
from peft import LoraConfig, get_peft_model, PeftModel
import torch
 
def get_llm(name, use_lora=False, lora_r=32, lora_alpha=2,finetune_llm=False,  torch_dtype=torch.float32):
    """Get LLM model and tokenizer based on model name.
    
    Args:
        name (str): Model name or path
        use_lora (bool): Whether to use LoRA
        lora_r (int): LoRA rank parameter
        lora_alpha (int): LoRA alpha parameter
        torch_dtype: Model dtype (default: torch.float32)
    """
    
    if "llama" in name.lower():
        llm_tokenizer = LlamaTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            use_fast=False
        )
        llm_model = LlamaForCausalLM.from_pretrained(
            name,
            torch_dtype=torch_dtype,
            device_map="auto"
        )
    
    elif "qwen" in name.lower():
        llm_tokenizer = Qwen2Tokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            use_fast=False
        )
        llm_model = Qwen2ForCausalLM.from_pretrained(
            name, 
            torch_dtype=torch_dtype,
            device_map="auto"
        )
    
    else:
        # Default to AutoTokenizer/AutoModel for other models
        llm_tokenizer = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            use_fast=False
        )
        llm_model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto"
        )

    for param in llm_model.parameters():
        param.requires_grad = finetune_llm

    if use_lora:
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules="all-linear",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )
        llm_model = get_peft_model(llm_model, peft_config)
        llm_model.print_trainable_parameters()


    return llm_tokenizer, llm_model


if __name__ == "__main__":
    models = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "/path/to/Qwen2.5-0.5B-Instruct",
        "meta-llama/Llama-2-7b-chat-hf"
    ]
    
    for model_name in models:
        try:
            tokenizer, model = get_llm(
                model_name, 
                use_lora=False,
                torch_dtype=torch.bfloat16
            )
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            print(f"Failed to load {model_name}: {str(e)}")