import argparse
import re
from ollama import chat, ChatResponse

def split_text(text, max_chars=5000):
    """
    Splits text into chunks without breaking sentences.
    Uses a max character limit instead of tokens.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split by sentence boundaries
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def ask(filename, ds_model, deepthink=True, print_log=True):
    """
    Processes a transcript file, splits it into chunks, and prompts DeepSeek via Ollama.

    :filename -> str: Path to the transcript file
    :ds_model -> str: DeepSeek model identifier (e.g., "deepseek-r1:14b")
    :deepthink -> bool: Whether to include additional thought processes in the final answer
    :print_log -> bool: Whether to print the final response to the console

    :return -> str: Final summarized meeting report
    """

    with open(filename, "r", encoding="utf-8", errors="replace") as file:
        text = file.read()

    chunks = split_text(text)
    partial_summaries = []

    for i, chunk in enumerate(chunks):
        system_prompt = f"""
        Внимательно изучи транскрипт записи встречи. Выяви участников встречи, основные тезисы встречи, запиши протокол встречи по следующему формату:
    
        1. 10 ключевых тезисов встречи
        2. Принятые решения, ответственные за их исполнения, сроки
        3. Ближайшие шаги. Отметь наиболее срочные задачи. Подробно опиши поставленные задачи каждому сотруднику, укажи сроки исполнения задач.

        Текст: {chunk}
        """

        response: ChatResponse = chat(  # Get the raw response from the model
            model=ds_model, messages=[
                {'role': 'user', 'content': system_prompt},
            ]
        )

        response_text = response.message['content']  # Extract response text
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
        model=ds_model, messages=[
            {'role': 'user', 'content': final_prompt}
        ]
    )

    final_response_text = final_response.message['content']

    if print_log:
        print(final_response_text)

    return final_response_text


# Command-line argument parsing
parser = argparse.ArgumentParser(description="User prompt")
parser.add_argument("input", type=str, help="Path to the transcript file")
args = parser.parse_args()

model = "deepseek-r1:14b"  # Specify the DeepSeek model
response = ask(args.input, model)
