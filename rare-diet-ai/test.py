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
        "ingredients": ["potato", "brown rice", "broccoli"]
    },
    "meal_type": "meal2",
    "consumed_so_far": {}
}

response = requests.post(url, json=payload)

print("Status code:", response.status_code)
print("Response:")
print(response.text)
