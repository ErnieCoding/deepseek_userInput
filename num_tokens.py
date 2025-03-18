import tiktoken

def count_tokens(filepath, encoding_name="cl100k_base"):
    with open(filepath, 'r', encoding="utf-8") as file:
        text = file.read()

    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)

    return len(tokens)


filepath = "tests/НД транскрипты/Командос17-12.txt"

print(f"File {filepath.split("/")[1]} has {count_tokens(filepath)} tokens")