import requests
import re
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3-coder:latest"

def call_ollama(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"]

def extract_read_file(text):
    match = re.search(r'<read_file>\s*path:\s*"([^"]+)"\s*</read_file>', text)
    if match:
        return match.group(1)
    return None

def read_file(path):
    if not os.path.exists(path):
        return f"[ERROR: File not found: {path}]"
    if os.path.isdir(path):
        return f"[ERROR: {path} is a directory]"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"[ERROR reading file: {e}]"

def run_agent(user_prompt):
    context = user_prompt

    for step in range(10):  # prevent infinite loops
        print(f"\n--- Step {step+1} ---")

        response = call_ollama(context)
        print(response)

        file_path = extract_read_file(response)

        if file_path:
            file_content = read_file(file_path)

            context += f"\n\n[FILE CONTENT: {file_path}]\n{file_content}\n"
        else:
            print("\n✅ Final Answer Reached")
            break


if __name__ == "__main__":
    prompt = "Summarise this project"
    run_agent(prompt)