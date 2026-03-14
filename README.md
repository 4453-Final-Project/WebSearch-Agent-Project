## 1) Create and activate a virtual environment
`python -m venv .venv`

### Activate venv (Windows):
`.venv\Scripts\activate`

### Activate venv (Mac/Linux):
`source .venv/bin/activate`
Confirm with: `which python'` before next step (should be .venv path).

## 2) Install the packages you need
`'pip install -U torch transformers accelerate huggingface_hub trl datasets peft`

## 3) Log into Hugging Face
`hf auth login`

## 4) Download the model locally
`hf download Qwen/Qwen3-3B --local-dir ./models/Qwen3-3B`

## 5) Create a quick test file called test_model.py with this inside:

```
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = "./models/Qwen3-3B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 6) Run the test
`python test_model.py` 

