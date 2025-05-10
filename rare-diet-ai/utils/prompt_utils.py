import json
from pathlib import Path

def load_disease_guide(path="data/disease_guide.json"):
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}

def generate_prompt(user_info, meal_type, disease_guide, consumed_so_far):
    disease_texts = []
    remaining_nutrients = {"protein": 0, "fat": 0, "carbohydrates": 0}

    # 질병별 제한 정보와 주의사항 정리
    for disease in user_info["disease"]:
        d_data = disease_guide.get(disease.lower())
        if not d_data:
            continue

        note = f"※ {d_data['note']}\n- 피해야 할 음식: {', '.join(d_data['avoid'])}\n- 권장 식품: {', '.join(d_data['safe'])}"
        disease_texts.append(note)

        limit = d_data.get("nutrition_limit", {})
        for nutrient in remaining_nutrients:
            remaining_nutrients[nutrient] += limit.get(nutrient, 0)

    # 소비한 영양소 차감
    for nutrient in remaining_nutrients:
        consumed = consumed_so_far.get(nutrient, 0)
        remaining_nutrients[nutrient] = max(remaining_nutrients[nutrient] - consumed, 0)

    # 명확한 JSON 포맷 요구 (medical_diets 형식)
    prompt = f"""
You are a professional clinical nutritionist. Please recommend a {meal_type.upper()} meal for the following user:

- Age: {user_info['age']}
- Gender: {user_info['gender']}
- Height: {user_info['height']}cm
- Allergies: {', '.join(user_info['allergy'])}
- Ingredients available: {', '.join(user_info['ingredients'])}

Health conditions:
{chr(10).join(disease_texts)}

Remaining daily nutrient limits:
- Protein: {remaining_nutrients['protein']}g
- Fat: {remaining_nutrients['fat']}g
- Carbohydrates: {remaining_nutrients['carbohydrates']}g

Please respond in the following strict JSON format:

{{
  "meal": {{
    "dish": "Dish Name",
    "menu": ["Ingredient 1 (amount)", "Ingredient 2 (amount)"],
    "notes": [
      "Note about allergy or disease",
      "Another important consideration"
    ],
    "calories": 000,
    "protein": 00,
    "carbs": 00,
    "fat": 00
  }}
}}
"""
    return prompt
