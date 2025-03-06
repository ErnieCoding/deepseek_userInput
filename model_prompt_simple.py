import argparse
import ollama
import time
from translate import translate_file

def ask(filename, model):
    # with open(filename, "r", encoding="utf-8", errors="replace") as file:
    #     text = file.read()
    
    translated_text = translate_file(filename)

    task = """Analyze the above transcript of 10 therapy sessions and generate structured notes for each session. Each session's notes should follow this format:\n\n
Session Summary – Key discussion points and main themes.\n
Patient Presentation – Emotional tone, engagement level, and any notable behaviors.\n
Therapist Interventions – Techniques used (e.g., CBT, active listening, Socratic questioning) and their impact.\n
Patient Responses & Insights – Significant reactions, breakthroughs, or challenges.\n
Recurring Patterns & Progress – Any themes, cognitive distortions, or behavioral patterns across sessions.\n
Future Considerations – Suggested focus areas for future therapy sessions."\n\n
Ensure that each session's notes maintain clarity, accuracy, and consistency in structure. If there are notable changes across sessions, highlight them in the final summary.
    """

    system_prompt = f"""{translated_text}\n
    Analyze the above transcript of 10 therapy sessions and generate structured notes for each session. Each session's notes should follow this format:\n\n
Session Summary – Key discussion points and main themes.\n
Patient Presentation – Emotional tone, engagement level, and any notable behaviors.\n
Therapist Interventions – Techniques used (e.g., CBT, active listening, Socratic questioning) and their impact.\n
Patient Responses & Insights – Significant reactions, breakthroughs, or challenges.\n
Recurring Patterns & Progress – Any themes, cognitive distortions, or behavioral patterns across sessions.\n
Future Considerations – Suggested focus areas for future therapy sessions."\n\n
Ensure that each session's notes maintain clarity, accuracy, and consistency in structure. If there are notable changes across sessions, highlight them in the final summary.
    """
    start = time.time()
    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': "You are an assistant, taking notes on Gestalt therapy transcripts."},
            {'role': 'user', 'content': system_prompt}
        ],
        stream=False,
    )

    response_text = response.get('message', {}).get('content', "Ошибка: ответ модели отсутствует.")

    end = time.time()

    with open("model_response.txt", "a", encoding="utf-8") as file:
        write_response = f"""-----------------------------------------------------------------------------------------------------------------------------------------------------\n\n
Модель: {model}\n
Транскрипт встречи: {filename} \n
Промпт:\n
<транскрипт>\n\n{task}\n
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
