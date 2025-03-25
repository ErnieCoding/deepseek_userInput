import tiktoken

def count_tokens(filepath = None, text = None, encoding_name="cl100k_base"):
    if filepath:
        with open(filepath, 'r', encoding="utf-8") as file:
            text = file.read()

    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)

    return len(tokens)