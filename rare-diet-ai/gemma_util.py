from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from huggingface_hub import login
import torch
import re
import json
import os

print("GCP environment")

# 모델 정보 및 디바이스 설정
MODEL_ID = "google/gemma-2b-it"
device = torch.device("cpu")

# Hugging Face 인증
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    try:
        login(hf_token)
        print("[INFO] Hugging Face login successful.")
    except Exception as e:
        print(f"[ERROR] Hugging Face login failed: {e}")
else:
    print("[WARNING] HUGGINGFACE_TOKEN environment variable not set.")

# 모델 로드
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=hf_token).to(device)
    generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=-1)  # CPU 사용 시 device=-1
    print("[INFO] Gemma model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    generator = None

# Gemma 호출 함수
import time
def call_gemma(prompt: str, max_tokens: int = 512) -> str:
    if not generator:
        return "[ERROR] Gemma model is not loaded."
    try:
        start = time.time()
        result = generator(prompt, max_new_tokens=max_tokens, temperature=0.7, do_sample=True)[0]["generated_text"]
        duration = time.time() - start
        print(f"[Gemma] Generated in {duration:.2f} seconds")
        return result
    except Exception as e:
        print(f"[ERROR] Gemma generation failed: {e}")
        print(f"[DEBUG] Prompt that caused failure:\n{prompt[:300]}...")
        return "{}"


def extract_json(text: str) -> dict | None:
    """
    Attempts to extract the largest valid JSON object from a text string.
    """
    stack = []
    start_idx = None

    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start_idx = i
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack:
                    candidate = text[start_idx:i+1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError as e:
                        print(f"[ERROR] JSON decode failed: {e}")
                        continue

    print("[ERROR] No valid JSON object found. Full text:")
    print(text)
    return None
