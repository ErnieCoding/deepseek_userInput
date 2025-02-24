import argparse
import re
from transformers import AutoTokenizer

#from ollama import Client

from ollama import chat, ChatResponse

#client = Client(host="http://127.0.0.1:11411")

tokenizer = AutoTokenizer.from_pretrained("gpt2", model_max_length=131072)

def split_by_tokens(text, max_tokens = 2000):
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]




def ask(filename, ds_model, deepthink=True, print_log=True):

    """
    Функция ask для промпта модели DeepSeek через ollama.

    :system_ptompt -> str: текстовый промпт для модели
    :ds_model -> str: нужная модель для тестирования в формате "deepseek-<model name>:<number of parameters> 
    :deepthink -> bool: включать ли порцию мышления в финальный ответ (необязательно) 
    :print_log -> bool: распечатать финальный ответ в консоль (необязательно)

    :return -> str, str: чистый текст без процесса мышления и процесс мышления отдельно от ответа
    """

    with open(filename, "r", encoding="utf-8", errors="replace") as file:
        text = file.read()

    chunks = split_by_tokens(text)
    partial_summaries = []

    for i, chunk in enumerate(chunks):
        system_prompt = """
        Внимательно изучи транскрипт записи встречи. Выяви участников встречи, основные тезисы встречи, запиши протокол встречи по следующему формату:
    
    1. 10 ключевых тезисов встречи
    2. Принятые решения, ответственные за их исполнения, сроки
    3. Ближайшие шаги. Отметь наиболее срочные задачи. Подробно опиши поставленные задачи каждому сотруднику, укажи сроки исполнения задач.

    Текст: {chunk}
        """

        response: ChatResponse = chat( # получить сырой ответ от модели
            model = ds_model, messages=[
                {'role':'user', 'content':system_prompt},
            ]
        )

        response_text = response['message']['content'] # вывод текстового марианта

        partial_summaries.append(response_text)

    final_prompt = f"""
    Ты — ассистент, анализирующий встречи. Ниже приведены анализы отдельных частей транскрипта.
    На их основе сделай единый итоговый отчет:

    {'\n\n'.join(partial_summaries)}

    Формат вывода:
    1. Участники встречи
    2. 10 ключевых тезисов
    3. Принятые решения (ответственные и сроки)
    4. Ближайшие шаги (сроки и исполнители)
    """

    final_response: ChatResponse = chat(
        model = ds_model, messages=[
            {'role':'user', 'content':final_prompt}
        ]
    )

    final_response_text = final_response['message']['content']

    if print_log:
        print(final_response_text)

    return clean, thinkProcess

# принять промпт в командной строке в текстовом формате
parser = argparse.ArgumentParser(description="User prompt")
parser.add_argument("input", type=str, help="The string to be used as prompt")
args = parser.parse_args()

model = "deepseek-r1:14b" # указать нужную модель
clean_text, think_process = ask(args.input, model)


