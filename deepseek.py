import argparse
import re

from ollama import chat
from ollama import ChatResponse

def ask(system_prompt, ds_model, deepthink=True, print_log=True):
    response: ChatResponse = chat(
        model = ds_model, messages=[
            {'role':'user', 'content':system_prompt},
        ]
    )

    response_text = response['message']['content']
    if print_log: print(response_text)

    thinkProcess = re.findall(r'<think>(.*?)</think>', response_text, flags=re.DOTALL)

    thinkProcess = "\n\n".join(thinkProcess).strip()

    clean = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

    return clean

parser = argparse.ArgumentParser(description="User prompt")
parser.add_argument("input", type=str, help="The string to be used as prompt")

args = parser.parse_args()
model = "deepseek-r1:14b"



response = ask(args.input, model)

print(response)

print('\n')
