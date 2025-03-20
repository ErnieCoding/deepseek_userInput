import requests
import json

def query_deepseek(filename, api_key):

    with open(filename, 'r', encoding="utf-8", errors="replace") as file:
        text = file.read()
    
    prompt = f"""
Внимательно изучи транскрипт записи встречи. Выяви участников встречи, основные тезисы встречи,
запиши протокол встречи на основе представленного транскрипта по следующему формату:
1. 10 ключевых тезисов встречи
2. Принятые решения, ответственные за их исполнения, сроки
3. Ближайшие шаги. Отметь наиболее срочные задачи Подробно опиши поставленные задачи каждому сотруднику, укажи сроки исполнения задач.

Транскрипт встречи: 
{text}

"""

    url = "https://api.deepseek.com/chat/completions"  
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1,
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code}, {response.text}"

if __name__ == "__main__":
    api_key = "sk-35982cadcb7e46389fbd548f6a211ccf"  # Replace with your API key
    user_prompt = input("Enter the filepath: ")
    response = query_deepseek(user_prompt, api_key)
    print("DeepSeek-R1 Response:", response)

