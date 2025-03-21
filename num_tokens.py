import tiktoken

def count_tokens(filepath = None, text = None, encoding_name="cl100k_base"):
    if filepath:
        with open(filepath, 'r', encoding="utf-8") as file:
            text = file.read()

    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)

    return len(tokens)


#filepath = "tests/НД транскрипты/Командос17-12.txt"

print(count_tokens(text="Summarize the following part of a meeting transcript. Extract key points, decisions made, and any assigned tasks with responsible people and deadlines."))