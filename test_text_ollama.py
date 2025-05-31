import ollama

def main():
    """
    Sends a prompt to the Qwen model via Ollama and prints the response.
    """
    model_name = "qwen2.5vl:3b"
    prompt_text = "What is the capital of France?"

    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt_text,
                },
            ]
        )
        # print(f"Prompt: {prompt_text}")
        print(f"Response: {response['message']['content']}")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure Ollama is running and the model is available.")

if __name__ == '__main__':
    main()
