# Repository Structure:

```
RL4453/
├── models/                     # local model weights (not tracked by git)
│   └── Qwen2.5-3B-Instruct/
├── .venv/                      # local Python environment
└── WebSearch-Agent-Project/    # this repository
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    ├── scripts/
    │   ├── download_models.sh
    │   ├── download_models.ps1
    │   └── test_model.py
    └── src/
        ├── agent/
        ├── training/
        ├── env/
        └── utils/
```

# Getting Started:
## 1) Create and activate a virtual environment
$ `python -m venv .venv`

### Activate venv (Windows):
$ `.venv\Scripts\activate`

### Activate venv (Mac/Linux):
$ `source .venv/bin/activate`

Confirm with: `which python` before next step (should be .venv path).

## 2) Install the packages you need
$ `pip install -r requirements.txt`

or

$ `pip install -U torch transformers accelerate huggingface_hub trl datasets peft`

## 3) Log into Hugging Face
You may need to create account if you haven't on [hugging face](https://huggingface.co), and create a access token (Profile > Access Tokens)

$ `hf auth login`

## 4) Download the model locally
Now for the model we have a couple options, but this is the reasoning. Qwen/Qwen3-<MODEL_SIZE> is already trained on web arena, but Qwen/Qwen2.5-<MODEL_SIZE> is not, so 2.5 is better for our purposes (as a baseline). Also, instruction fine tuning is important for our purposes (web search), as such we will be using: 

[Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

With that step 4 is to download the `Qwen2.5-3B-Instruct` model from hf:

$ `hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/Qwen2.5-3B-Instruct`


## 5) Run a quick test
Assuming you have downloaded the correct model run this to test:

$ `python scripts/test_model.py` 

