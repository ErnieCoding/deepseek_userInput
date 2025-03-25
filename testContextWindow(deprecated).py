#from num_tokens import count_tokens
import ollama
from libreTranslateFile import translate_to_eng
import argparse

def test_context_repeating(model, num_lines, num_recall):
    text_arr = []
    for i in range(1, num_lines+1):
        text_arr.append(f"Line {i}: This is a test line with ID {i % 10}TS{i}.")
    
    text = "\n".join(text_arr)

    prompt = f"Here's a long text. Repeat the first {num_recall} lines exactly as they appeared:\n\n {text}\n\n "

    response = ollama.chat(
        model=model,
        messages=[{"role":"user", "content":prompt}],
        options={"temperature":0.1, "num_ctx":131072}
    )

    response_text = response.get('message', {}).get('content', "Ошибка: ответ модели отсутствует.")

    with open("test.txt", "w", encoding="utf-8") as file:
        file.write(text)

    expected = "\n".join(text.split("\n")[:num_recall])

    return {
        "model_response": response_text,
        "expected_output": expected,
    }

def test_context_transcript(filepath, model):
    with open(filepath, "r", encoding="utf-8") as file:
        transcript = file.read()

    transcript = translate_to_eng(transcript)

    with open("test.txt", "w", encoding="utf-8") as file:
        file.write(transcript)
    
    prompt = f"From the following transcript, repeat the first 3 lines exactly as they appear: {transcript}"

    response = ollama.chat(
        model=model,
        messages=[{"role":"user", "content":prompt}],
        options={"temperature":0.1, "num_ctx":131072}
    )
    
    response_text = response.get('message', {}).get('content', "Ошибка: ответ модели отсутствует.")

    return response_text

if __name__ == "__main__":
    result = test_context_repeating("llama3.1:8b", 10000, 30)

    print("Expected Output:\n", result["expected_output"])
    print("Model Output:\n", result["model_response"])

    print(test_context_transcript(input("Enter transcipt's filepath: "), "llama3.1:8b-instruct-fp16"))