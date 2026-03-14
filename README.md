# Repository Structure:

```
RL4453/
├── models/                     # MAKE SURE THAT THIS IS A SIBLING DIRECTORY OF THE REPO
│   └── Qwen2.5-3B-Instruct/    # THE MODEL
├── .venv/                      # MAKE SURE THAT THIS IS A SIBLING DIRECTORY OF THE REPO
└── WebSearch-Agent-Project/    # THE REPO
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
From here make sure that you are in some directory that is NOT the git repo, you will clone it later on in these steps when you run the model test.

## 1) Create and activate a virtual environment
$ `python -m venv .venv`

### Activate venv (Windows):
$ `.venv\Scripts\activate`

### Activate venv (Mac/Linux):
$ `source .venv/bin/activate`

Confirm with: `which python` before next step (should be .venv path).

## 2) Install the packages you need
$ `pip install -r requirements.txt` // the below option is more reliable (I haven't tested requirements.txt yet, but we'll include this in submission)

or

$ `pip install -U torch transformers accelerate huggingface_hub trl datasets peft`

## 3) Log into Hugging Face
You may need to create account if you haven't on [hugging face](https://huggingface.co), and create a access token (Profile > Access Tokens)

$ `hf auth login`

## 4) Download the model locally
Now for the model we have a couple options, but this is the reasoning. Qwen/Qwen3-<MODEL_SIZE> is already trained on web arena, but Qwen/Qwen2.5-<MODEL_SIZE> is not, so 2.5 is better for our purposes (as a baseline). Also, instruction fine tuning is important for our purposes (web search), as such we will be using: 

[Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

With that, step 4 is to download the `Qwen2.5-3B-Instruct` model from hf:

$ `hf download Qwen/Qwen2.5-3B-Instruct --local-dir ./models/Qwen2.5-3B-Instruct`


## 5) Run a quick test
Assuming you have downloaded the correct model run this to test:

### Clone
```
$ git clone git@github.com:4453-Final-Project/WebSearch-Agent-Project.git

or

$ git clone https://github.com/4453-Final-Project/WebSearch-Agent-Project.git
```

### Run test
$ `python scripts/test_model.py` 

This will take some (possibly alot of) time to run locally.

### Example test output:

```
.../asn/WebSearch-Agent-Project on main  λ python scripts/test_model.py
General Specification/Expected output:
Load the local tokenizer and model, run one short generation, and print the decoded result.
User prompt: Hello
Model path: /mnt/c/Users/shonh/RL4453/asn/models/Qwen2.5-3B-Instruct
Status: loading tokenizer...
Status: tokenizer loaded in 1.1s
Status: loading model weights...
Loading weights: 100%|████████████████████████████████████████████████████████████████| 434/434 [00:00<00:00, 965.12it/s]
Status: model loaded in 2.1s
Status: tokenizing prompt...
Status: generating output...
Status: generation finished in 48.5s
Model Output: Hello, I'm trying to find a solution for the following problem. I have a dataset and I want
Total time: 51.8s
```
