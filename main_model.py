import ollama
import os
import sys
import argparse

# The only commands a robot may output
VALID_COMMANDS = {'move forward', 'turn left', 'turn right', 'stop'}


def build_prompt() -> str:
    """
    Returns the fixed prompt instructing the vision-LLM to output
    exactly one valid robot movement command.
    """
    return (
        "You are a robot that can see using a camera.\n"
        "Your tasks is to move around the world.\n"
        "DO NOT collide with any objects.\n"
        "Stay away from walls.\n"
        "Strictly only output one of the following commands:\n"
        "'move forward', 'turn left', 'turn right', 'stop'"
    )


def get_robot_command(image_path: str, model_name: str = "qwen2.5vl:3b") -> str:
    """
    Sends the image and prompt to the specified multimodal Ollama model
    and returns a single validated movement command.
    Raises ValueError if the model replies with an unexpected string.
    """
    prompt_text = build_prompt()
    print(f"Using model: {model_name}")
    print(f"Sending image: {image_path}")

    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt_text,
                "images": [image_path],
            }
        ],
    )

    # Normalise and validate response
    command = response["message"]["content"].strip().lower()
    print(f"Model raw response: {command}")

    if command not in VALID_COMMANDS:
        raise ValueError(
            f"Model returned invalid command '{command}'. "
            f"Expected one of {', '.join(sorted(VALID_COMMANDS))}."
        )

    return command


def main() -> None:
    """
    CLI entry-point. Example:

        python main_model.py -i image1.jpg -m qwen2.5vl:3b
    """
    parser = argparse.ArgumentParser(description="Generate a robot action from an image.")
    parser.add_argument(
        "-i",
        "--image",
        default="image1.jpg",
        help="Path to the input image (default: %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="qwen2.5vl:3b",
        help="Ollama model name to use (default: %(default)s)",
    )
    args = parser.parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}", file=sys.stderr)
        sys.exit(1)

    try:
        command = get_robot_command(image_path, args.model)
        print(f"\nRobot command: {command}")
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        print(
            "Please ensure Ollama is running and the specified model supports images.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()