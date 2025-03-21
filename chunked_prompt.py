import argparse
import ollama
import time
from libreTranslateFile import translate_to_eng, translate_to_rus
import re

system_role = "You are a business assistant that creates structured meeting reports based on meeting transcripts. Extract key discussion points, decisions (with responsible parties and deadlines), and next steps. Ensure clarity, accuracy, and relevance."

chunk_summary_prompt = "Summarize the following part of a meeting transcript. Extract key points, decisions made, and any assigned tasks with responsible people and deadlines."

rus_final_prompt = """Внимательно изучи транскрипт записи встречи. Выяви участников встречи, основные тезисы встречи, запиши протокол встречи на основе представленного транскрипта по следующему формату:
1. 10 ключевых тезисов встречи
2. Принятые решения, ответственные за их исполнения, сроки
3. Ближайшие шаги. Отметь наиболее срочные задачи Подробно опиши поставленные задачи каждому сотруднику, укажи сроки исполнения задач.
"""
translated_prompt = translate_to_eng(rus_final_prompt)

def chunk_text(text, max_length=8000):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length <= max_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def get_model_response(prompt, transcript, model):
    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': system_role},
            {'role': 'user', 'content': f"{prompt}\n{transcript}"}
        ],
        stream=False,
        options={'temperature': 0.3, 'num_ctx': 8192},
    )
    return response.get('message', {}).get('content', "Ошибка: ответ модели отсутствует.")

def ru_response(args, model):
    with open(args.filename, 'r', encoding='utf-8', errors='replace') as file:
        transcript = file.read()
    
    transcript = translate_to_eng(transcript)
    chunks = chunk_text(transcript)
    summarized_chunks = []
    
    for chunk in chunks:
        summary = get_model_response(chunk_summary_prompt, chunk, model)
        summarized_chunks.append(summary)
    
    full_summary = " ".join(summarized_chunks)
    
    final_prompt = f"""{translated_prompt}"""
    
    final_response = get_model_response(final_prompt, full_summary, model)
    translated_response = translate_to_rus(final_response)
    
    # Save test results into a txt file
    test_dir = "tests/"
    model_prefix = {
        "llama3.1:8b": "llama_test/",
        "llama3.1:8b-instruct-fp16": "llama_test/",
        "qwen2.5:14b": "qwen_test/",
        "qwen2.5:14b-instruct-fp16": "qwen_test/",
        "mistral-nemo:12b": "mistral_test/",
        "mistral-nemo:12b-instruct-2407-fp16": "mistral_test/",
        "gemma3:27b":"gemma_test/",
    }.get(model, "deepseek_test/")
    
    test_dir += model_prefix
    base_filename = args.filename.split("/")[-1].split(".")[0]
    is_instruct = "instruct" in model
    model_type = "instruct" if is_instruct else "base"
    result_file = f"{test_dir}{base_filename}_{model_type}_model_response.txt"
    
    with open(result_file, "w", encoding="utf-8") as file:
        file.write(f"""
Модель: {model}\n
Транскрипт встречи: {args.filename.split('/')[-1]} \n
Роль: {system_role}\n

Финальный отчет:\n
{translated_response}\n
-----------------------------------------------------------------------------------------------------------------------------------------------------\n
""")
    return final_summary, translated_response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="User prompt")
    parser.add_argument("filename", type=str, help="Name of the txt file to be analyzed")

    # Model selection arguments
    parser.add_argument("--llama", action="store_true", help="output a llama3.1:8b model response")
    parser.add_argument("--llama_instruct", action="store_true", help="output a llama-instruct model response")
    parser.add_argument("--qwen", action="store_true", help="output a qwen model response")
    parser.add_argument("--qwen_instruct", action="store_true", help="output a qwen-instruct model response")
    parser.add_argument("--mistral", action="store_true", help="output a mistral 12b model response")
    parser.add_argument("--mistral_instruct", action="store_true", help="output a mistral-instruct model response")
    parser.add_argument("--deepseek", action="store_true", help="output a deepseek 14b response")
    parser.add_argument("--deepseek_distill", action="store_true", help="output a deepseek qwen distill response")
    parser.add_argument("--gemma", action="store_true", help="output a gemma3 model response")
    
    args = parser.parse_args()

    if args.llama_instruct:
        model = "llama3.1:8b-instruct-fp16"
    elif args.llama:
        model = "llama3.1:8b"
    elif args.qwen:
        model = "qwen2.5:14b"
    elif args.qwen_instruct:
        model = "qwen2.5:14b-instruct-fp16"
    elif args.mistral:
        model = "mistral-nemo:12b"
    elif args.mistral_instruct:
        model = "mistral-nemo:12b-instruct-2407-fp16"
    elif args.deepseek:
        model = "deepseek-r1:14b"
    elif args.deepseek_distill:
        model = "deepseek-r1:14b-qwen-distill-fp16"
    elif args.gemma:
        model = "gemma3:27b"
    
    ru_response(args, model)
