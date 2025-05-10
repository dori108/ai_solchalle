from keybert import KeyBERT
import nltk
from nltk.tokenize import sent_tokenize
from nutrition_fetcher import get_nutrition_from_openfoodfacts

nltk.download("punkt", quiet=True)
kw_model = KeyBERT()

def extract_keywords_from_diet_text(text):
    """
    식단 설명 텍스트에서 문장 단위로 키워드 추출.
    긍정/부정 의미로 분류하여 추천 식재료와 피해야 할 식재료 목록 반환
    """
    sentences = sent_tokenize(text)
    positive_keywords = set()
    negative_keywords = set()

    for sent in sentences:
        keywords = kw_model.extract_keywords(sent, top_n=3, stop_words="english")
        keyword_list = [kw[0] for kw in keywords]
        lowered = sent.lower()

        if any(term in lowered for term in ["avoid", "not recommended", "dangerous", "do not eat", "should not eat"]):
            negative_keywords.update(keyword_list)
        elif any(term in lowered for term in ["recommended", "good choice", "should eat", "beneficial", "healthy", "safe"]):
            positive_keywords.update(keyword_list)

    return {
        "recommended": list(positive_keywords),
        "to_avoid": list(negative_keywords)
    }

def analyze_diet_nutrition_by_keywords(keywords_dict):
    """
    추천된 키워드들에 대해 OpenFoodFacts 또는 USDA에서 영양 정보를 조회
    """
    result = {}
    for keyword in keywords_dict.get("recommended", []):
        result[keyword] = get_nutrition_from_openfoodfacts(keyword)
    return result

def detect_conflicts(keywords_dict, allergies, diseases, disease_guide):
    """
    추천/비추천 키워드가 사용자의 알레르기나 질병 제한과 충돌하는지 확인
    """
    all_keywords = keywords_dict.get("recommended", []) + keywords_dict.get("to_avoid", [])
    conflicts = set()

    # 알레르기와의 충돌
    for allergy in allergies:
        for keyword in all_keywords:
            if allergy.lower() in keyword.lower():
                conflicts.add(keyword)

    # 질병 제한과의 충돌
    for disease in diseases:
        d_info = disease_guide.get(disease.lower())
        if not d_info:
            continue
        for forbidden in d_info.get("avoid", []):
            for keyword in all_keywords:
                if forbidden.lower() in keyword.lower():
                    conflicts.add(keyword)

    return list(conflicts)
