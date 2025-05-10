import os
import re
import json
import requests
from pathlib import Path
import tempfile

# 경로 설정
DISEASE_LIMIT_PATH = "data/disease_limit.json"
LOG_PATH = os.path.join(tempfile.gettempdir(), "logs")  # Render 호환
os.makedirs(LOG_PATH, exist_ok=True)

# disease_limit.json 로딩
def load_disease_limits():
    if Path(DISEASE_LIMIT_PATH).exists():
        with open(DISEASE_LIMIT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []

# disease_limit에서 해당 질병 정보 찾기
def find_disease_info(name):
    limits = load_disease_limits()
    name_lower = name.lower()
    for item in limits:
        if item["diseaseName"].lower() == name_lower:
            return {
                "avoid": [],
                "safe": [],
                "nutrition_limit": {
                    "protein": item.get("proteinLimit", 0),
                    "carbohydrates": item.get("sugarLimit", 0),
                    "fat": 0  # sodiumLimit은 제외
                },
                "note": item.get("notes", "")
            }
    return None

# PubMed에서 질병 관련 식이 논문 검색
def fetch_pubmed_abstracts(disease_name):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    term = disease_name.replace(" ", "+") + "+diet+restriction"
    search_url = f"{base_url}esearch.fcgi?db=pubmed&term={term}&retmax=3&retmode=json"

    try:
        res = requests.get(search_url, timeout=10)
        res.raise_for_status()
        ids = res.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return "", []
    except Exception as e:
        print(f"[ERROR] PubMed 검색 실패: {str(e)}")
        return "", []

    abstracts = []
    for pmid in ids:
        fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={pmid}&retmode=text&rettype=abstract"
        try:
            res = requests.get(fetch_url, timeout=10)
            if res.ok:
                abstracts.append(res.text.strip())
        except:
            continue

    return "\n\n".join(abstracts), ids

# 논문에서 식이 관련 정보 추출 (간단한 규칙 기반)
def extract_diet_info_rule_based(text):
    avoid = []
    safe = []
    note = text[:500] + "..." if text else ""
    nutrition_limit = {"protein": 0, "carbohydrates": 0, "fat": 0}

    lowered = text.lower()
    if "low-protein" in lowered:
        avoid.append("high-protein food")
        safe.append("low-protein alternatives")
        nutrition_limit["protein"] = 0.5
    if "cornstarch" in lowered:
        safe.append("uncooked cornstarch")
        nutrition_limit["carbohydrates"] = 100
    if "fat-restricted" in lowered:
        avoid.append("high-fat food")
        nutrition_limit["fat"] = 10

    return avoid, safe, nutrition_limit, note

# 최종 질병 정보 처리
def process_disease(disease_name):
    print(f"[INFO] 질병 정보 수집 시도: {disease_name}")

    # 1단계: disease_limit 확인
    info = find_disease_info(disease_name)
    if info:
        print(f"[INFO] disease_limit에서 '{disease_name}' 정보 발견")
        return info

    # 2단계: PubMed 검색 시도
    text, ids = fetch_pubmed_abstracts(disease_name)
    if not text:
        print(f"[WARNING] '{disease_name}'에 대한 PubMed 검색 실패 또는 결과 없음.")
        return {
            "avoid": [],
            "safe": [],
            "nutrition_limit": {},
            "note": "정보 없음"
        }

    avoid, safe, limit, note = extract_diet_info_rule_based(text)

    # 로그 저장
    log_file = os.path.join(LOG_PATH, f"{disease_name.lower().replace(' ', '_')}_abstract.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(text)

    return {
        "avoid": avoid,
        "safe": safe,
        "nutrition_limit": limit,
        "note": note
    }