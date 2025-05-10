from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import torch
import re
import json

MODEL_ID = "google/gemma-2b-it"
device = torch.device("cpu")

# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
generator = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=0)

def call_gemma(prompt: str, max_tokens: int = 512) -> str:
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
