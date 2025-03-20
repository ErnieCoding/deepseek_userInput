import ollama
import argparse
import libreTranslateFile
from pydantic import BaseModel
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

class OutputFormat(BaseModel):
    key_points: list[str]
    decisions_made: list[dict]
    tasks: list[dict]

def split_text(txt):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[".", "\n"],
        chunk_size = 3072,
        chunk_overlap = 0,
    )

    chunks = text_splitter.split_text(txt)
    return chunks

def create_vector_store(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def retrieve_relevant_chunks(vector_store, query, top_k=50):
    docs = vector_store.similarity_search(query, k=top_k)
    return "\n".join([doc.page_content for doc in docs])

def prompt_model(args, model):
    with open(args.filename, "r", encoding='utf-8') as file:
        transcript = file.read()
    
    transcript = libreTranslateFile.translate_to_eng(transcript)

    chunks = split_text(transcript)
    vector_store = create_vector_store(chunks)

    query = "Summarize the key points, decisions made, and next steps from this meeting."
    relevant_text = retrieve_relevant_chunks(vector_store, query)

    prompt = f"""Carefully study the transcript excerpts below and generate a meeting report with the following format:
    1. 10 Key Points of the Meeting
    2. Decisions made, those responsible for their implementation, and deadlines.
    3. Next steps, marking the most urgent tasks and describing them in detail.
    
    Transcript Excerpts:
    {relevant_text}
"""
    response = ollama.chat(
        model=model,
        messages=[{"role":"user", "content":prompt}],
        format=OutputFormat.model_json_schema(),
        options={"temperature":0.1, "num_ctx":131072}
    )

    summary = OutputFormat.model_validate_json(response.message.content)

    with open("test.txt", "w", encoding="utf-8") as file:
        file.write(str(summary.key_points))
        file.write(str(summary.decisions_made))
        file.write(str(summary.tasks))

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="User prompt")
    parser.add_argument("filename", type=str, help="Name of the txt file to be analyzed")

    # arguments for model outputs
    parser.add_argument("--llama", action="store_true", help="output a llama3.1:8b model response") # option to engage base llama model
    parser.add_argument("--llama_instruct", action="store_true", help="output a llama-instruct model response") # option to engage llama-instruct
    parser.add_argument("--qwen", action="store_true", help="output a qwen model response") # option to engage base qwen2.5 model
    parser.add_argument("--qwen_instruct", action="store_true", help="output a qwen-instruct model response") # option to engage qwen_instruct
    parser.add_argument("--mistral", action="store_true", help="output a mistral 12b model response") # option to engange base mistral model
    parser.add_argument("--mistral_instruct", action="store_true", help="output a mistral-instruct model response") # option to engange mistral-instruct model
    parser.add_argument("--deepseek", action="store_true", help="output a deepseek 14b response")
    parser.add_argument("--deepseek_distill", action="store_true", help="output a deepseek qwen distill response")
    
    args = parser.parse_args()

    if args.llama_instruct:
        model = "llama3.1:8b-instruct-fp16"
    elif args.llama:
        model = "llama3.1:8b"
    elif args.qwen:
        model = "qwen2.5:14b"
    elif args.qwen_instruct:
        model = "qwen2.5:14b-instruct-fp16"
    elif args.mistral:
        model = "mistral-nemo:12b"
    elif args.mistral_instruct:
        model = "mistral-nemo:12b-instruct-2407-fp16"
    elif args.deepseek:
        model = "deepseek-r1:14b"
    elif args.deepseek_distill:
        model = "deepseek-r1:14b-qwen-distill-fp16"

    response = prompt_model(args, model)
    print(response)