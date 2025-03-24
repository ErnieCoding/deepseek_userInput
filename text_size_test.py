import pypdf
import ollama
from tokenCounter import count_tokens
import re

def extract(filepath, num_tokens):
    """
    Extract part of a pdf text based on the number of tokens being tested
    """
    reader = pypdf.PdfReader(filepath)
    pages = reader.pages

    extracted_text = ""
    for page in pages:
        curr_text = page.extract_text()
        sentences = re.split(r'(?<=[.!?])\s+', curr_text)
        for sentence in sentences:
            sentence_len = count_tokens(text=sentence)
            if count_tokens(text=extracted_text) + sentence_len <= num_tokens:
                extracted_text += sentence
            else:
                return extracted_text.strip()
    
    return extracted_text.strip()

def determine_numctx(model:str) -> int:
    """
    Determine the maximum num_ctx depending on the model used
    """
    pass

def get_summary(text:str, prompt:str, model:str) -> str:
    """
    Prompt the model based on the provided chunk of a text.
    """
    model_response = ollama.chat(
        model=model,
        messages=[
            {'role':'user', 'content':f'{prompt}\n{text}'}
        ],
        stream=False,
        options={'temperature':0, 'num_ctx': determine_numctx(model)}
    )
    
    return model_response.get('message', {}).get('content', "Error: no response from the model.")

#TODO: implement test file recording based on the model
def record_test(text, prompt, model, response):
    """
    Records test in a txt file with text, prompt, model, and model's response
    """
    pass

#TODO: implement main wrapper and unittest for each model
text_from_book = extract("tests/text_size_test/Rye.pdf", 8192)

print(text_from_book + "\n\n")

print(count_tokens(text=text_from_book))
