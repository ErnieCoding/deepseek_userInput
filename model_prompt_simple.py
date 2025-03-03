import argparse
import ollama
import time

def ask(filename, model):
    start = time.time()
    with open(filename, "r", encoding="utf-8", errors="replace") as file:
        text = file.read()
    
    system_prompt = """Ты - ассистент, анализирующий встречи и составляющий протоколы по предоставленному формату.

Проанализируй транскрипт записи встречи и подготовь отчет в следующем формате:

1. **10 ключевых тезисов встречи**  
   - Кратко и четко изложи основные идеи, обсуждавшиеся в ходе встречи.

2. **Принятые решения**  
   - Опиши решения, к которым пришли участники.  
   - Укажи ответственных за исполнение.  
   - Добавь сроки выполнения.

3. **Ближайшие шаги**  
   - Укажи наиболее срочные задачи.  
   - Подробно опиши задачи каждого сотрудника и сроки их выполнения.

Важно! Ответ должен строго следовать указанному формату.

Теперь обработай следующий транскрипт:
"""

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
Транскрипт встречи: {filename.split("/")[1]} \n\n
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

model = "qwen2.5:14b-instruct-fp16"
response = ask(args.input, model)
