import argparse

from ollama import chat, ChatResponse

def ask(filename, model):

    with open(filename, 'r', encoding="utf-8", errors="replace") as file:
        text = file.read()
    
    system_prompt = f"""
Внимательно изучи транскрипт записи встречи. Выяви участников встречи, основные тезисы встречи,
запиши протокол встречи на основе представленного транскрипта по следующему формату:
1. 10 ключевых тезисов встречи
2. Принятые решения, ответственные за их исполнения, сроки
3. Ближайшие шаги. Отметь наиболее срочные задачи Подробно опиши поставленные задачи каждому сотруднику, укажи сроки исполнения задач.

Транскрипт встречи: 
{text}

"""
    response: ChatResponse = chat(
        model = model, messages=[
            {'role':'user', 'content':system_prompt},
        ]
    )

    response_text = response['message']['content']

    print(response_text)

    return response_text

parser = argparse.ArgumentParser(description='User prompt')
parser.add_argument("input", type=str, help="The string to be used as prompt")
args = parser.parse_args()

model = "deepseek-r1:14b"
response = ask(args.input, model)