import requests

url = "https://ai-solchalle-chocosongi.onrender.com/generate_diet"

payload = {
    "user_info": {
        "age": 25,
        "gender": "female",
        "height": 160,
        "weight": 50,
        "allergy": [],
        "disease": ["PKU"],
        "ingredients": ["감자", "현미", "브로콜리"]
    },
    "meal_type": "meal2",
    "consumed_so_far": {}
}

response = requests.post(url, json=payload)

print("응답 코드:", response.status_code)
print("응답 내용:")
print(response.text)  # JSON이 아니더라도 안전하게 출력
