import argparse
import re

#from ollama import Client

from ollama import chat, ChatResponse

#client = Client(host="http://127.0.0.1:11411")


def ask(system_prompt, ds_model, deepthink=True, print_log=True):

    """
    Функция ask для промпта модели DeepSeek через ollama.

    :system_ptompt -> str: текстовый промпт для модели
    :ds_model -> str: нужная модель для тестирования в формате "deepseek-<model name>:<number of parameters> 
    :deepthink -> bool: включать ли порцию мышления в финальный ответ (необязательно) 
    :print_log -> bool: распечатать финальный ответ в консоль (необязательно)

    :return -> str, str: чистый текст без процесса мышления и процесс мышления отдельно от ответа
    """
    response: ChatResponse = chat( # получить сырой ответ от модели
        model = ds_model, messages=[
            {'role':'user', 'content':system_prompt},
        ]
    )

    response_text = response['message']['content'] # вывод текстового марианта

    thinkProcess = re.findall(r'<think>(.*?)</think>', response_text, flags=re.DOTALL) # выделить процесс мышления из ответа

    thinkProcess = "\n\n".join(thinkProcess).strip()

    clean = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip() # исключить процесс мышления из финального ответа

    if print_log: # выдача чистого ответа в командном терминале 
        print(clean)

    return clean, thinkProcess

# принять промпт в командной строке в текстовом формате
parser = argparse.ArgumentParser(description="User prompt")
parser.add_argument("input", type=str, help="The string to be used as prompt")
args = parser.parse_args()

model = "deepseek-r1:14b" # указать нужную модель
clean_text, think_process = ask(args.input, model)


