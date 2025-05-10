import os
import json
import requests
from pathlib import Path

USDA_PATH = "data/usda/"

# ✅ USDA 백업 JSON에서 영양 성분 조회
def search_usda_backup_data(query):
    folder = Path(USDA_PATH)
    if not folder.exists():
        return None

    nutrients_map = {
        "protein": "Protein",
        "fat": "Total lipid (fat)",
        "carbohydrates": "Carbohydrate, by difference",
        "calories": "Energy"
    }

    for file in folder.glob("*.json"):
        try:
            with open(file, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue

        for item in data.get("FoundationFoods", []):
            food_name = item.get("description", "").lower()
            if query.lower() in food_name:
                result = {k: "정보 없음" for k in nutrients_map}
                for fn in item.get("foodNutrients", []):
                    n_name = fn.get("nutrient", {}).get("name", "")
                    for key, expected in nutrients_map.items():
                        if expected.lower() == n_name.lower():
                            result[key] = fn.get("amount", "정보 없음")
                result["name"] = item.get("description", query)
                return result
    return None

# ✅ OpenFoodFacts 우선, 실패 시 USDA 백업으로 영양 정보 조회
def get_nutrition_from_openfoodfacts(query):
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    params = {
        "search_terms": query,
        "json": 1,
        "search_simple": 1,
        "action": "process"
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if data.get("count", 0) > 0 and data.get("products"):
            p = data["products"][0]
            return {
                "name": p.get("product_name", query),
                "calories": p.get("nutriments", {}).get("energy-kcal_100g", "알 수 없음"),
                "protein": p.get("nutriments", {}).get("proteins_100g", "알 수 없음"),
                "fat": p.get("nutriments", {}).get("fat_100g", "알 수 없음"),
                "carbohydrates": p.get("nutriments", {}).get("carbohydrates_100g", "알 수 없음")
            }
    except Exception as e:
        print(f"[⚠️] OpenFoodFacts 조회 실패 for '{query}': {str(e)}")

    # ⛑ 백업 사용
    fallback = search_usda_backup_data(query)
    if fallback:
        print(f"[🔄] USDA 백업 사용: '{query}'")
        return fallback

    # ❌ 조회 실패
    return {
        "name": query,
        "calories": "정보 없음",
        "protein": "정보 없음",
        "fat": "정보 없음",
        "carbohydrates": "정보 없음"
    }
