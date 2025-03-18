#from num_tokens import count_tokens
import ollama
import argparse

def test_context_repeating(model, num_lines, num_recall):
    text_arr = []
    for i in range(1, num_lines+1):
        text_arr.append(f"Line {i}: This is a test line with ID {i % 10}TS{i}.")
    
    text = "\n".join(text_arr)

    prompt = f"Here's a long text. \n\n {text}\n\n Repeat the first {num_recall} lines exactly as they appeared:"

    response = ollama.chat(
        model=model,
        messages=[{"role":"user", "content":prompt}],
        options={"temperature":0.3}
    )

    response_text = response.get('message', {}).get('content', "Ошибка: ответ модели отсутствует.")

    expected = "\n".join(text.split("\n")[:num_recall])

    #print(text)

    return {
        "model_response": response_text,
        "expected_output": expected,
    }

def test_context_transcript(filepath, model):
    pass


result = test_context_repeating("llama3.1:8b", 500000, 30)

print("Expected Output:\n", result["expected_output"])
print("Model Output:\n", result["model_response"])