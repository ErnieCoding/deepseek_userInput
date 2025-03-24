#TODO: try passing tools into the chat using the example from https://github.com/ollama/ollama-python/blob/main/examples/tools.py
import argparse
import ollama
import time
from libreTranslateFile import translate_to_eng, translate_to_rus


system_role = "You are a business assistant that creates structured meeting reports based on meeting transcripts. Extract key discussion points, decisions (with responsible parties and deadlines), and next steps. Ensure clarity, accuracy, and relevance."

meeting_context= """Контекст встречи: Регулярная встреча команды продаж стартапа RConf по результатам выполнения задач на неделю, статусу пилотных проектов у клиентов, организационным вопросам. 
Стартап RConf предоставляет клиентам платформу видеоконфренцсвязи с AI для анализа рабочих встреч, оценки и развития сотрудников.

Участники встречи (могут присутствовать не все участники):
Алексей Воронин- руководитель стартапа
Артем Садыков- руководитель пилотных проектов
Александр Швецов- менеджер продаж
Максим Перфильев- руководитель R&D
Елена Евтеева- проджект-менеджер
Татьяна Вавилова- управляющий партнер, специалист по HR"""

rus_prompt = f"""Внимательно изучи транскрипт записи встречи. Выяви участников встречи, основные тезисы встречи, запиши протокол встречи на основе представленного транскрипта по следующему формату:
1. 10 ключевых тезисов встречи
2. Принятые решения, ответственные за их исполнения, сроки
3. Ближайшие шаги. Отметь наиболее срочные задачи Подробно опиши поставленные задачи каждому сотруднику, укажи сроки исполнения задач.

{meeting_context}
    """

prompt_translated = translate_to_eng(rus_prompt)

#print(prompt_translated) # проверка перевода промпта

# Get response in russian (translated response)
def ru_response(args, model):
    with open(args.filename, 'r', encoding='utf-8', errors='replace') as file:
        transcript = file.read()

    transcript = translate_to_eng(transcript)
  
    system_prompt = f"""{prompt_translated}\n
{transcript}
    """

    start = time.time()

    # Get model response through ollama
    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': system_role},
            {'role': 'user', 'content': system_prompt}
        ],
        stream=False,
        options={'temperature': 0.3, 'num_ctx': 131072},
    )

    response_text = response.get('message', {}).get('content', "Ошибка: ответ модели отсутствует.")

    end = time.time()

    translated_response = translate_to_rus(response_text)


    # Save test results into a txt file
    test_dir = "tests/"

    model_prefix = {
    "llama3.1:8b": "llama_test/",
    "llama3.1:8b-instruct-fp16": "llama_test/",
    "qwen2.5:14b": "qwen_test/",
    "qwen2.5:14b-instruct-fp16": "qwen_test/",
    "mistral-nemo:12b": "mistral_test/",
    "mistral-nemo:12b-instruct-2407-fp16": "mistral_test/"}.get(model, "deepseek_test/")

    test_dir += model_prefix

    base_filename = args.filename.split("/")[2].split(".")[0]

    is_instruct = "instruct" in model
    model_type = "instruct" if is_instruct else "base"

    result_file = f"{test_dir}{base_filename}_{model_type}_model_response.txt"

    with open(result_file, "w", encoding="utf-8") as file:
        write_response = f"""Модель: {model}\n
Транскрипт встречи: {args.filename.split("/")[2]} \n
Роль: {system_role}
Промпт:\n
{prompt_translated}\n<транскрипт>\n\n
Ответ модели:\n
{translated_response}\n\n
Время ответа: {end - start:.2f} sec
-----------------------------------------------------------------------------------------------------------------------------------------------------\n\n
"""
        file.write(write_response)

    #print(response_text)
    return response_text, translated_response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="User prompt")
    parser.add_argument("filename", type=str, help="Name of the txt file to be analyzed")

    # arguments for model outputs
    parser.add_argument("--llama", action="store_true", help="output a llama3.1:8b model response") # option to engage base llama model
    parser.add_argument("--llama_instruct", action="store_true", help="output a llama-instruct model response") # option to engage llama-instruct
    parser.add_argument("--qwen", action="store_true", help="output a qwen model response") # option to engage base qwen2.5 model
    parser.add_argument("--qwen_instruct", action="store_true", help="output a qwen-instruct model response") # option to engage qwen_instruct
    parser.add_argument("--mistral", action="store_true", help="output a mistral 12b model response") # option to engange base mistral model
    parser.add_argument("--mistral_instruct", action="store_true", help="output a mistral-instruct model response") # option to engange mistral-instruct model
    parser.add_argument("--deepseek", action="store_true", help="output a deepseek 14b response")
    parser.add_argument("--deepseek_distill", action="store_true", help="output a deepseek qwen distill response")
    
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

    response = ru_response(args, model)


# Get response in english
# def eng_response(filename, model):
#     with open(filename, "r", encoding="utf-8", errors="replace") as file:
#         text = file.read()
    
#     translated_text = translate_to_eng(text)

#     system_prompt = f"""{translated_text}\n
#     Analyze the above transcript of 10 therapy sessions and generate structured notes for each session. Each session's notes should follow this format:\n\n
# Session Summary – Key discussion points and main themes.\n
# Patient Presentation – Emotional tone, engagement level, and any notable behaviors.\n
# Therapist Interventions – Techniques used (e.g., CBT, active listening, Socratic questioning) and their impact.\n
# Patient Responses & Insights – Significant reactions, breakthroughs, or challenges.\n
# Recurring Patterns & Progress – Any themes, cognitive distortions, or behavioral patterns across sessions.\n
# Future Considerations – Suggested focus areas for future therapy sessions."\n\n
# Ensure that each session's notes maintain clarity, accuracy, and consistency in structure. If there are notable changes across sessions, highlight them in the final summary.
#     """

#     start = time.time()

#     response = ollama.chat(
#         model=model,
#         messages=[
#             {'role': 'system', 'content': system_role},
#             {'role': 'user', 'content': system_prompt}
#         ],
#         stream=False,
#     )

#     response_text = response.get('message', {}).get('content', "Ошибка: ответ модели отсутствует.")

#     end = time.time()

#     with open("model_response_ENG.txt", "a", encoding="utf-8") as file:
#         write_response = f"""-----------------------------------------------------------------------------------------------------------------------------------------------------\n\n
# Модель: {model}\n
# Транскрипт встречи: {filename} \n
# Промпт:\n
# <транскрипт>\n\n{task_eng}\n
# Ответ модели:\n
# {response_text}\n\n

# Время ответа: {end - start:.2f} sec
# -----------------------------------------------------------------------------------------------------------------------------------------------------\n\n
# """
#         file.write(write_response)

#     #print(response_text)
#     return response_text
