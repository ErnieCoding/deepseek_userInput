import argparse
import ollama
import time

def ask(filename, model):
    start = time.time()
    with open(filename, "r", encoding="utf-8", errors="replace") as file:
        text = file.read()
    
    system_prompt = """You are a specialist in taking notes on Gestalt therapy texts Carefully study the attached text and provide notes for sessions 8 and 9."""

    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': text}
        ],
        stream=False,
    )

    response_text = response.get('message', {}).get('content', "Ошибка: ответ модели отсутствует.")

    end = time.time()

    with open("model_response.txt", "a", encoding="utf-8") as file:
        write_response = f"""-----------------------------------------------------------------------------------------------------------------------------------------------------\n\n
Модель: {model}\n
Транскрипт встречи: {filename} \n\n
Промпт:\n
{system_prompt}\n
Ответ модели:\n
{response_text}\n\n

Время ответа: {end - start:.2f} sec
-----------------------------------------------------------------------------------------------------------------------------------------------------\n\n
"""
        file.write(write_response)

    print(response_text)
    return response_text

parser = argparse.ArgumentParser(description="User prompt")
parser.add_argument("input", type=str, help="The string to be used as prompt")
args = parser.parse_args()

model = "deepseek-r1:14b"
response = ask(args.input, model)
