import ollama
import os

def main():
    """
    Sends a prompt and an image to a Qwen multimodal model via Ollama 
    and prints the response.
    """
    model_name = "qwen2.5vl:3b"
    prompt_text = """
    You are a robot that can see using a camera.
    Your tasks is to move around the world.
    DO NOT collide with any objects.
    Stay away from walls.
    Strictly only output one of the following commands:
    'move forward', 'turn left', 'turn right', 'stop'
    """
    
    image_path = "./image1.jpg"

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        print("Please update the 'image_path' variable in the script.")
        return

    try:
        print(f"Using model: {model_name}")
        print(f"Loading image from: {image_path}")
        
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt_text,
                    'images': [image_path]  # Pass the image path here
                },
            ]
        )
        print(f"\nPrompt: {prompt_text}")
        print(f"Response: {response['message']['content']}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure Ollama is running, the specified model supports images,")

if __name__ == '__main__':
    main()
