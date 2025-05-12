from flask import Flask, request, jsonify
from pathlib import Path
import json
import random
import os
from pubmed_fetcher import process_disease
from gemma_util import call_gemma, extract_json
from diet_generator import extract_keywords_from_diet_text, analyze_diet_nutrition_by_keywords, detect_conflicts

from huggingface_hub import login

hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    login(hf_token)
else:
    print("[WARNING] Hugging Face token is not set. Gemma may not work.")


app = Flask(__name__)

DISEASE_LIMIT_PATH = "data/disease_limit.json"
FALLBACK_DIETS_PATH = "data/medical_diets.json"
REFERENCE_WEIGHT = 55

def load_disease_limits():
    if Path(DISEASE_LIMIT_PATH).exists():
        with open(DISEASE_LIMIT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []

def load_fallback_diets():
    if Path(FALLBACK_DIETS_PATH).exists():
        with open(FALLBACK_DIETS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}

def scale_diet(meal, user_weight):
    try:
        scale_factor = user_weight / REFERENCE_WEIGHT
        return {
            "dish": meal["dish"],
            "menu": meal["menu"],
            "notes": meal["notes"],
            "calories": round(meal["calories"] * scale_factor),
            "protein": round(meal["protein"] * scale_factor, 1),
            "carbs": round(meal["carbs"] * scale_factor, 1),
            "fat": round(meal["fat"] * scale_factor, 1),
        }
    except Exception as e:
        print(f"[ERROR] fallback scaling failed: {e}")
        return None

def is_invalid_diet(parsed: dict, meal_key: str) -> bool:
    try:
        meal = parsed.get(meal_key, {})
        required_fields = ["dish", "menu", "notes", "calories", "protein", "carbs", "fat"]
        for field in required_fields:
            value = meal.get(field)
            if value in [None, "...", "정보 없음", "", []]:
                return True
        return False
    except:
        return True

def generate_prompt(user_info, meal_type, disease_info, consumed_so_far):
    disease_texts = []
    remaining_nutrients = {"protein": 0, "fat": 0, "carbohydrates": 0, "sodium": 0}

    for d in user_info["disease"]:
        d_data = disease_info.get(d.lower())
        if not d_data:
            continue
        disease_texts.append(f"* {d_data['note']}\n- Avoid: {', '.join(d_data['avoid'])}\n- Safe: {', '.join(d_data['safe'])}")
        limit = d_data.get("nutrition_limit", {})
        for k in remaining_nutrients:
            if k in limit:
                remaining_nutrients[k] += limit[k]

    for k in remaining_nutrients:
        consumed = consumed_so_far.get(k, 0)
        remaining_nutrients[k] = max(remaining_nutrients[k] - consumed, 0)

    prompt = f"""
You are a professional nutritionist. Recommend a {meal_type.upper()} meal for the following user:
- Age: {user_info['age']}
- Gender: {user_info['gender']}
- Height: {user_info['height']}cm
- Weight: {user_info['weight']}kg
- Ingredients available: {', '.join(user_info['ingredients'])}

Health notes:
{chr(10).join(disease_texts)}

Remaining daily intake allowance:
- Protein: {remaining_nutrients['protein']}g
- Fat: {remaining_nutrients['fat']}g
- Carbohydrates: {remaining_nutrients['carbohydrates']}g
- Sodium: {remaining_nutrients['sodium']}mg

Please respond in JSON format only:
{{
  "meal": {{
    "dish": "...",
    "menu": ["..."],
    "notes": ["..."],
    "calories": 0,
    "protein": 0,
    "carbs": 0,
    "fat": 0
  }}
}}
"""
    return prompt

@app.route("/generate_diet", methods=["POST"])
def generate_diet():
    data = request.json
    user = data["user_info"]
    diseases = user.get("disease", [])
    meal_type = data.get("meal_type", "breakfast").lower()
    consumed = data.get("consumed_so_far", {})

    disease_info = {}
    for d in diseases:
        disease_info[d.lower()] = process_disease(d)

    for d in diseases:
        d_info = disease_info.get(d.lower(), {})
        if "nutrition_limit" not in d_info or not d_info["nutrition_limit"]:
            disease_info[d.lower()] = {
                "avoid": [],
                "safe": [],
                "nutrition_limit": {
                    "protein": user.get("protein", 0),
                    "carbohydrates": user.get("sugar", 0),
                    "fat": 0,
                    "sodium": user.get("sodium", 0)
                },
                "note": "Based on user-provided daily nutrient limits."
            }

    prompt = generate_prompt(user, meal_type, disease_info, consumed)
    result = call_gemma(prompt)
    parsed = extract_json(result)

    fallback_used = False

    if not parsed or "meal" not in parsed or is_invalid_diet(parsed, "meal"):
        print("[Fallback] Gemma output is incomplete or invalid. Using fallback.")
        fallback_diets = load_fallback_diets()
        for d in diseases:
            d_key = d.replace(" ", "_").replace("(", "").replace(")", "") + "_meals"
            if d_key in fallback_diets:
                meals = fallback_diets[d_key]
                meal = meals.get("meal1")
                scaled = scale_diet(meal, user["weight"])
                if scaled:
                    parsed = {"meal": scaled}
                    fallback_used = True
                    print(f"[Fallback success] Used: {d_key}")
                    break

        if not parsed:
            print("[Retry] Fallback failed, retrying Gemma")
            result = call_gemma(prompt)
            parsed = extract_json(result)

    keywords = extract_keywords_from_diet_text(json.dumps(parsed))
    nutrition = analyze_diet_nutrition_by_keywords(keywords)
    conflicts = detect_conflicts(keywords, user.get("allergy", []), diseases, disease_info)

    if "meal" in parsed:
        if "notes" not in parsed["meal"]:
            parsed["meal"]["notes"] = []
        if conflicts:
            parsed["meal"]["notes"].append(
                f"\u26a0\ufe0f This meal may conflict with your conditions: {', '.join(conflicts)}"
            )

    return jsonify({
        "diet": parsed,
        "nutrition": nutrition,
        "conflicts": conflicts,
        "fallback_used": fallback_used
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
