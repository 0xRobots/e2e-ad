# e2e-ad
Vision Language Action Model for End-to-End Autonomous Driving

# Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

# Serve Model using ollama
```bash
ollama serve qwen2.5vl:3b
```

# Setup
```bash
pip install ollama
```

# Actions

- Move Forward
- Stop
- Turn Left
- Turn Right

## Usage

### 1. Single-image inference

```bash
python main_model.py -i image1.jpg -m qwen2.5vl:3b
```

The script prints one **validated** robot command, e.g.

```
Robot command: move forward
```

### 3. Quick tests

```bash
python test_image_ollama.py     # single image test
python test_text_ollama.py      # prompt-only test
```

### 4. Valid commands

The model **must** reply with exactly one of:

- move forward  
- turn left  
- turn right  
- stop  

Treat any other response as invalid.

## Troubleshooting

- Make sure Ollama is running: `ollama serve`
- If the model is missing, pull it first: `ollama pull qwen2.5vl:3b`
- To force GPU usage: `export OLLAMA_FORCE_CUDA=1`

---
