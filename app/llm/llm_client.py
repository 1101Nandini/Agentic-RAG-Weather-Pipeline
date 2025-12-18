"""
LLM Client 
----------------------------------------------
- Uses GPU automatically if available
- Falls back to CPU if not
- Supports gated Hugging Face models
"""
import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from huggingface_hub import login

load_dotenv()

_LLM = None

def get_llm():
    global _LLM
    if _LLM is not None:
        return _LLM

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # Device Setup
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    
    print(f"🚀 Loading LLM on {device.upper()}...")

    model_name = os.getenv("HF_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if use_gpu else torch.float32,
        low_cpu_mem_usage=True,
        device_map=device 
    )

    # Pipeline
    text_generation_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,        # Greedy decoding (Strict facts)
        repetition_penalty=1.1, # Prevents looping
        return_full_text=False, # Don't return the prompt in the output
        pad_token_id=tokenizer.eos_token_id
    )

    _LLM = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return _LLM