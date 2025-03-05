from deep_translator import DeeplTranslator
import argparse
#import asyncio


def translate_file(filename):
    
    with open(filename, "r", encoding="utf-8", errors="replace") as file:
        text = file.read()
    
    chunk_size = 4500
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    translated_text = []
    for chunk in chunks:
        try:
            translated_chunk = DeeplTranslator(api_key="2bf1e3f6-4e16-40a9-a997-9a30c2d6ab46:fx", source='ru', target='en', use_free_api=True).translate(chunk)
            translated_text.append(translated_chunk)
        except Exception as e:
            print(f"Error translating chunk: {e}")

    full_translation = "\n".join(translated_text)

    # with open("translated_text_test.txt", "w", encoding="utf-8") as file:
    #     file.write(full_translation)
    return full_translation

parser = argparse.ArgumentParser(description="File name")
parser.add_argument("filename", type=str, help="File to translate")
args = parser.parse_args()

translated = translate_file(args.filename)