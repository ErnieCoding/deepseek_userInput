import pypdf
from langchain_ollama import ChatOllama
from tokenCounter import count_tokens
import re

def extract(filepath, num_tokens):
    """
    Extracts part of a pdf text based on the number of tokens being tested
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

def get_summary(text, prompt, model):
    """
    Prompts the model based on the provided chunk of a text.
    """

    pass



