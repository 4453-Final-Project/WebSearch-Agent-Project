## 1) Create and activate a virtual environment
$ `python -m venv .venv`

### Activate venv (Windows):
$ `.venv\Scripts\activate`

### Activate venv (Mac/Linux):
$ `source .venv/bin/activate`

Confirm with: `which python` before next step (should be .venv path).

## 2) Install the packages you need
$ `pip install -U torch transformers accelerate huggingface_hub trl datasets peft`

## 3) Log into Hugging Face
You may need to create account if you haven't on [hugging face](https://huggingface.co), and create a access token (Profile > Access Tokens)

$ `hf auth login`

## 4) Download the model locally
Now for the model we have a couple options, but this is the reasoning. Qwen/Qwen3-<MODEL_SIZE> is already trained on web arena, but Qwen/Qwen2.5-<MODEL_SIZE> is not, so 2.5 is better for our purposes (as a baseline). Also, instruction fine tuning is important for our purposes (web search), as such we will be using: 

[Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

With that step 4 is to download the `Qwen2.5-3B-Instruct` model from hf:

$ `hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/Qwen2.5-3B-Instruct`

## 5) Quick Test 
Create a python script called `test_model.py` with this inside:

```
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = "./models/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 6) Run the test
$ `python test_model.py` 

