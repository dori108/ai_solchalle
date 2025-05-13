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
            if field not in meal:
                print(f"[DEBUG] Missing field: {field}")
                return True

        if not isinstance(meal["dish"], str) or len(meal["dish"].strip()) < 2:
            print("[DEBUG] Invalid dish name")
            return True

        if not isinstance(meal["menu"], list) or len(meal["menu"]) == 0:
            print("[DEBUG] Invalid menu list")
            return True

        if not isinstance(meal["notes"], list):
            print("[DEBUG] Notes is not a list")
            return True

        for field in ["calories", "protein", "carbs", "fat"]:
            value = meal[field]
            if not isinstance(value, (int, float)) or value <= 0:
                print(f"[DEBUG] Invalid value for {field}: {value}")
                return True

        return False
    except Exception as e:
        print(f"[ERROR] Exception in is_invalid_diet: {e}")
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
You are a professional nutritionist and your task is to recommend a {meal_type.upper()} meal for a specific user.
You will be provided with health details, dietary restrictions, and ingredient availability.
 Your response must ONLY be a well-formatted JSON object following the exact structure shown below.

User Information:
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

RESPONSE INSTRUCTIONS — PLEASE FOLLOW STRICTLY:
- You MUST respond with ONLY valid **JSON**.
- DO NOT add any extra text, explanation, comment, greeting, or closing.
- DO NOT omit or rename any keys.
- Every field shown below must be included — even if the value is 0 or an empty string ("").
- Do not use ellipses (...), placeholders, or incomplete values.
- If a value is unknown, write 0 (for numbers) or "" (for strings) — do not leave out any field.

 REQUIRED JSON FORMAT (copy exactly):
{{
  "meal": {{
    "dish": "Grilled Chicken Salad",
    "menu": ["Grilled chicken breast", "Mixed greens", "Cherry tomatoes", "Olive oil dressing"],
    "notes": ["Low-carb, high-protein meal suitable for most dietary restrictions."],
    "calories": 350,
    "protein": 32.5,
    "carbs": 15.0,
    "fat": 12.0
  }}
}}

Now respond with the meal plan in EXACTLY the same JSON format shown above.
"""

    return prompt

@app.route("/generate_diet", methods=["POST"])
def generate_diet():
    from random import choice, shuffle

    try:
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
        fallback_reason = ""

        if not parsed or "meal" not in parsed or is_invalid_diet(parsed, "meal"):
            print("[Fallback] Gemma output invalid. Trying fallback.")
            fallback_diets = load_fallback_diets()
            fallback_found = False

            for d in diseases:
                d_key = d.replace(" ", "_").replace("(", "").replace(")", "").lower() + "_meals"
                if d_key not in fallback_diets:
                    print(f"[Fallback] No fallback meals found for key: {d_key}")
                    continue

                meals = fallback_diets[d_key]
                meal_keys = list(meals.keys())
                shuffle(meal_keys)

                for key in meal_keys:
                    raw_meal = meals[key]
                    scaled = scale_diet(raw_meal, user["weight"])
                    if scaled and not is_invalid_diet({"meal": scaled}, "meal"):
                        parsed = {"meal": scaled}
                        fallback_used = True
                        fallback_reason = f"Gemma failed. Fallback used: {d_key} / {key}"
                        fallback_found = True
                        print(f"[Fallback success] Used {d_key} / {key}")
                        break
                    else:
                        print(f"[Fallback] Skipped invalid or unscalable fallback: {d_key} / {key}")

                if fallback_found:
                    break

            if not fallback_found:
                print("[Fallback failed] No valid fallback meals found. Using default safe meal.")
                parsed = {
                    "meal": {
                        "dish": "Basic Rice Bowl",
                        "menu": ["Steamed rice", "Boiled chicken", "Blanched spinach"],
                        "notes": ["⚠️ Gemma and fallback both failed. This is a safe default meal."],
                        "calories": 400,
                        "protein": 25,
                        "carbs": 45,
                        "fat": 10
                    }
                }
                fallback_used = True
                fallback_reason = "Gemma and all fallback meals failed. Default safe meal used."

        keywords = extract_keywords_from_diet_text(json.dumps(parsed))
        nutrition = analyze_diet_nutrition_by_keywords(keywords)
        conflicts = detect_conflicts(keywords, diseases, disease_info)

        if "meal" in parsed:
            if "notes" not in parsed["meal"]:
                parsed["meal"]["notes"] = []
            if conflicts:
                parsed["meal"]["notes"].append(
                    f"⚠️ This meal may conflict with your conditions: {', '.join(conflicts)}"
                )
            if fallback_used:
                parsed["meal"]["notes"].append(f"⚠️ Fallback used. Reason: {fallback_reason}")

        return jsonify({
            "diet": parsed,
            "nutrition": nutrition,
            "conflicts": conflicts,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason if fallback_used else None
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
