from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from huggingface_hub import login
import torch
import re
import json
import os

# ✅ 환경 변수에서 Hugging Face 토큰 불러오기
hf_token = os.environ.get("HF_TOKEN")  # Render에서는 이 이름 사용!
if hf_token:
    login(token=hf_token)
    print("[INFO] Hugging Face 로그인 성공")
else:
    print("[WARNING] HF_TOKEN 환경변수가 설정되지 않았습니다. 모델 로딩에 실패할 수 있습니다.")

MODEL_ID = "google/gemma-2b-it"
device = torch.device("cpu")

# ✅ 모델과 토크나이저 로드
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
    generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=0)
except Exception as e:
    print(f"[ERROR] 모델 로딩 실패: {e}")
    generator = None

def call_gemma(prompt: str, max_tokens: int = 512) -> str:
    if not generator:
        return "[ERROR] Gemma 모델이 로드되지 않았습니다."
    result = generator(prompt, max_new_tokens=max_tokens, temperature=0.7, do_sample=True)[0]["generated_text"]
    return result

def extract_json(text: str) -> dict:
    match = re.search(r"\{[\s\S]+?\}", text)
    if not match:
        print("[ERROR] JSON 형식이 감지되지 않았습니다. 원문 출력:")
        print(text)
        return {}

    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 파싱 실패: {e}")
        print("[원문]:")
        print(match.group())

        # 💡 보정 시도: 마지막 괄호가 부족한 경우
        fixed = match.group().strip()
        while not fixed.endswith("}"):
            fixed += "}"
        try:
            return json.loads(fixed)
        except:
            return {}
