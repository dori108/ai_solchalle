from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import torch
import re
import json
import os

print("[INFO] Hugging Face login 생략 (GCP 환경)")

MODEL_ID = "google/gemma-2b-it"
device = torch.device("cpu")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=device)
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    generator = None

def call_gemma(prompt: str, max_tokens: int = 512) -> str:
    if not generator:
        return "[ERROR] Gemma model is not loaded."
    result = generator(prompt, max_new_tokens=max_tokens, temperature=0.7, do_sample=True)[0]["generated_text"]
    return result

def extract_json(text: str) -> dict:
    match = re.search(r"\{[\s\S]+?\}", text)
    if not match:
        print("[ERROR] JSON format not found. Original output:")
        print(text)
        return {}

    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        fixed = match.group().strip()
        while not fixed.endswith("}"):
            fixed += "}"
        try:
            return json.loads(fixed)
        except:
            return {}
