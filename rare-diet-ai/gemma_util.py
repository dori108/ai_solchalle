from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from huggingface_hub import login
import torch
import re
import json
import os

# ✅ 환경 변수에서 Hugging Face 토큰 불러오기
hf_token = os.environ.get("HF_TOKEN")  # Hugging Face Spaces나 Render에서 환경변수로 등록 필요
if hf_token:
    login(token=hf_token)
    print("[INFO] Hugging Face login succeeded")
else:
    print("[WARNING] HF_TOKEN is not set. Model may not load.")
    generator = None  # 명확히 설정

MODEL_ID = "google/gemma-2b-it"
device = torch.device("cpu")

# ✅ 모델과 토크나이저 로드
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=device)
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    generator = None

# ✅ Gemma 호출 함수
def call_gemma(prompt: str, max_tokens: int = 512) -> str:
    if not generator:
        return "[ERROR] Gemma model is not loaded."
    result = generator(prompt, max_new_tokens=max_tokens, temperature=0.7, do_sample=True)[0]["generated_text"]
    return result

# ✅ JSON 파싱 함수
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
        print("[Original text]:")
        print(match.group())

        # 💡 괄호 누락 보정 시도
        fixed = match.group().strip()
        while not fixed.endswith("}"):
            fixed += "}"
        try:
            return json.loads(fixed)
        except:
            return {}
