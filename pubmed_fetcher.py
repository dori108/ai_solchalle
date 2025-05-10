import os
import re
import json
import requests
from pathlib import Path

DISEASE_LIMIT_PATH = "data/disease_limit.json"
LOG_PATH = "logs"
Path(LOG_PATH).mkdir(exist_ok=True)

def load_disease_limits():
    if Path(DISEASE_LIMIT_PATH).exists():
        with open(DISEASE_LIMIT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []

def find_disease_info(name):
    limits = load_disease_limits()
    name_lower = name.lower()
    for item in limits:
        if item["diseaseName"].lower() == name_lower:
            # disease_guide 대체 구조 생성
            return {
                "avoid": [],
                "safe": [],
                "nutrition_limit": {
                    "protein": item.get("proteinLimit", 0),
                    "carbohydrates": item.get("sugarLimit", 0),
                    "fat": 0  # sodiumLimit은 제외, fat 정보 없음
                },
                "note": item.get("notes", "")
            }
    return None

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

def process_disease(disease_name):
    print(f"[INFO] 질병 정보 수집 시도: {disease_name}")

    info = find_disease_info(disease_name)
    if info:
        print(f"[INFO] disease_limit에서 '{disease_name}' 정보 발견")
        return info

    # fallback to PubMed
    text, ids = fetch_pubmed_abstracts(disease_name)
    if not text:
        print(f"[WARNING] '{disease_name}'에 대한 PubMed 검색 실패 또는 결과 없음.")
        return {
            "avoid": [], "safe": [], "nutrition_limit": {}, "note": "정보 없음"
        }

    avoid, safe, limit, note = extract_diet_info_rule_based(text)

    # 로그 기록
    log_file = f"{LOG_PATH}/{disease_name.lower().replace(' ', '_')}_abstract.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(text)

    return {
        "avoid": avoid,
        "safe": safe,
        "nutrition_limit": limit,
        "note": note
    }

if __name__ == "__main__":
    result = process_disease("phenylketonuria")
    print(json.dumps(result, indent=2, ensure_ascii=False))