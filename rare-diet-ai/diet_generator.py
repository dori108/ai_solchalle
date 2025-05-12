from keybert import KeyBERT
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nutrition_fetcher import get_nutrition_from_openfoodfacts
import difflib

nltk.download("punkt", quiet=True)
kw_model = KeyBERT()

def extract_keywords_from_diet_text(text):
    """
    Extract keywords sentence by sentence and classify them based on sentiment.
    Return recommended and to_avoid keyword lists.
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
    Fetch nutritional info for recommended keywords.
    """
    result = {}
    for keyword in keywords_dict.get("recommended", []):
        result[keyword] = get_nutrition_from_openfoodfacts(keyword)
    return result

def fuzzy_match(keyword, target_list, threshold=0.8):
    """
    Check if a keyword closely matches any item in a target list using fuzzy ratio.
    """
    for target in target_list:
        ratio = difflib.SequenceMatcher(None, keyword.lower(), target.lower()).ratio()
        if ratio >= threshold:
            return True
    return False

def detect_conflicts(keywords_dict, allergies, diseases, disease_guide):
    """
    Detect conflicts between extracted keywords and user allergies or disease restrictions.
    """
    all_keywords = keywords_dict.get("recommended", []) + keywords_dict.get("to_avoid", [])
    conflicts = set()

    # Allergy conflict
    for allergy in allergies:
        for keyword in all_keywords:
            if fuzzy_match(keyword, [allergy]):
                conflicts.add(keyword)

    # Disease-related food restrictions
    for disease in diseases:
        d_info = disease_guide.get(disease.lower())
        if not d_info:
            continue
        for forbidden in d_info.get("avoid", []):
            for keyword in all_keywords:
                if fuzzy_match(keyword, [forbidden]):
                    conflicts.add(keyword)

    return list(conflicts)
