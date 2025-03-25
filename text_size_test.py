import pypdf
import ollama
import pypdf.errors
from tokenCounter import count_tokens
import re
from modelinfo import get_context_length
import argparse

def extract(filepath:str, num_tokens:int, prompt:str) -> str:
    """
    Extract part of a pdf text based on the number of tokens being tested
    """
    try:
        reader = pypdf.PdfReader(filepath)
        pages = reader.pages

        prompt_length = count_tokens(text=prompt)
        extracted_text = ""

        if num_tokens <= prompt_length:
            raise ValueError("Number of tokens must be greater than the prompt length.")

        for page in pages:
            curr_text = page.extract_text()

            if not curr_text:
                print("No text found on this page.") 
                continue

            curr_text = curr_text.encode('utf-8', 'replace').decode('utf-8')

            sentences = re.split(r'(?<=[.!?])\s+', curr_text)
            for sentence in sentences:
                sentence_len = count_tokens(text=sentence)
                if (prompt_length + count_tokens(text=extracted_text)) + sentence_len <= num_tokens:
                    extracted_text += sentence
                else:
                    return extracted_text.strip()
    except FileNotFoundError:
        print("Error: PDF file not found.")
        return ""
    except pypdf.errors.PdfReadError as e:
        print(f"Error while reading file: {e}")
        return ""
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def get_summary(text:str, prompt:str, model:str) -> str:
    """
    Prompt the model based on the provided chunk of a text.
    """
    try:
        model_response = ollama.chat(
            model=model,
            messages=[
                {'role':'user', 'content':f'{prompt}\n{text}'}
            ],
            stream=False,
            options={'temperature':0, 'num_ctx': get_context_length(model)}
        )
        
        return model_response.get('message', {}).get('content', "Error: no response from the model.")
    except Exception as e:
        return f"Error: {str(e)}"

def record_test(text:str, prompt:str, model:str, response:str) -> None:
    """
    Records test in a txt file with text, prompt, model, and model's response
    """

    test_dir = "tests/text_size_test/"

    model_prefix = {
        "llama3.1:8b": "llama",
        "llama3.1:8b-instruct-fp16": "llama",
        "qwen2.5:14b": "qwen",
        "qwen2.5:14b-instruct-fp16": "qwen",
        "mistral-nemo:12b": "mistral",
        "mistral-nemo:12b-instruct-2407-fp16": "mistral",
        "gemma3:27b":"gemma",
        "phi4:14b":"phi4",
        "deepseek-r1:14b":"deepseek",
        "deepseek-r1:14b-qwen-distill-fp16":"deepseek",
    }.get(model, "unknown")

    test_dir += model_prefix
    is_instruct = "instruct" in model
    model_type = "instruct" if is_instruct else "base"
    text_size = count_tokens(text=text) + count_tokens(text=prompt)
    result_file = f"{test_dir}_{model_type}_{text_size}_modelresponse.txt"

    with open(result_file, "w", encoding="utf-8", errors="replace") as file:
        file.write(f"""Model: {model}\n\n
Prompt: {prompt}\n
Text size (with prompt): {text_size}\n
Model Response: {response}\n\n
Text: \n{text}\n\n
""")

# Parsing args and recording tests
if __name__ == "__main__":
    prompt = """Conduct a holistic, precision-driven text comprehension analysis. Your goal is to maintain 100% information integrity while preserving exact contextual nuances.

Deliver:

1. An exhaustive factual summary capturing all key events and details without distortion or omission.
2. A thematic analysis exploring core ideas and recurring motifs.
3.A deep symbolic/metaphorical interpretation supported by textual evidence.
4. Extraction of five pivotal narrative components with justification.

Map out with full detail:

1. All character interactions and relationships.
2. The complete plot progression with no missing elements.
3. Narrative techniques, including structural choices, perspective, and literary devices.
4. A linguistic breakdown covering syntax, diction, tone, narrative voice, and literary techniques.
"""

    parser = argparse.ArgumentParser(description="User input")
    parser.add_argument("num_tokens", type=int, help="Number of tokens to process at a time")
    parser.add_argument("filepath", type=str, help="Filepath to pdf")
    args = parser.parse_args()

    response: ollama.ListResponse = ollama.list()
    for model in response.models:
        model_name = model.model

        if model_name == "nomic-embed-text:latest":
            continue
        
        extracted_text = extract(args.filepath, args.num_tokens, prompt)

        summary = get_summary(extracted_text, prompt, model_name)

        record_test(extracted_text, prompt, model_name, summary)