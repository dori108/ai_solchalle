from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from huggingface_hub import login
import torch
import re
import json
import os

# ✅ 환경 변수에서 Hugging Face 토큰 불러오기 (HF_TOKEN을 권장 이름으로 사용)
hf_token = os.environ.get("HF_TOKEN")  # Render 환경 변수 설정에서 이 이름 사용!
if hf_token:
    login(token=hf_token)
    print("[INFO] Hugging Face login")
else:
    print("[WARNING] HF_TOKEN is not defined")

MODEL_ID = "google/gemma-2b-it"
device = torch.device("cpu")

# ✅ 모델과 토크나이저 로드
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=0)
except Exception as e:
    print(f"[ERROR] failed to load model: {e}")
    generator = None

# ✅ Gemma 호출 함수
def call_gemma(prompt: str, max_tokens: int = 512) -> str:
    if not generator:
        return "[ERROR] Gemma is not loaded."
    result = generator(prompt, max_new_tokens=max_tokens, temperature=0.7, do_sample=True)[0]["generated_text"]
    return result

# ✅ JSON 파싱 함수
def extract_json(text: str) -> dict:
    match = re.search(r"\{[\s\S]+?\}", text)
    if not match:
        print("[ERROR] JSON form is not found. original note:")
        print(text)
        return {}

    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing fail: {e}")
        print("[원문]:")
        print(match.group())

        # 💡 괄호 누락 보정 시도
        fixed = match.group().strip()
        while not fixed.endswith("}"):
            fixed += "}"
        try:
            return json.loads(fixed)
        except:
            return {}
