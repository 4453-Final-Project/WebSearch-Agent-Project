## Overview
This repository is the implementation workspace for the RL course project. The current Task 1 goal is simple: make the local runtime, GPTQ model path, and BrowserGym/WebArena connectivity reproducible with one verification script.

The expected workspace layout is:

```text
asn/
|- .venv/
|- models/
|  `- Qwen2.5-3B-Instruct-GPTQ-Int4/
`- WebSearch-Agent-Project/
```

Use Python 3.12. BrowserGym/WebArena is sensitive to interpreter and browser setup, so do not swap versions casually.

## Setup
From `WebSearch-Agent-Project/`:

```bash
source ../.venv/bin/activate
python --version
which python
```

Expected:

```text
Python 3.12.x
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is stale, the fallback install is:

```bash
pip install -U torch transformers accelerate huggingface_hub trl datasets peft browsergym browsergym-webarena playwright optimum gptqmodel
```

## Download The Model
Task 1 targets the GPTQ checkpoint, not the full-precision checkpoint:

- model: `Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4`
- local path: `../models/Qwen2.5-3B-Instruct-GPTQ-Int4`

Download it directly:

```bash
hf download Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 --local-dir ../models/Qwen2.5-3B-Instruct-GPTQ-Int4
```

Or use the helper scripts:

```bash
bash scripts/download_models.sh
```

On Windows PowerShell:

```powershell
.\scripts\download_models.ps1
```

## Configure WebArena URLs
Copy the example file and replace the placeholder hostname with your WebArena host:

```bash
cp .env.example .env
```

Then source it into your shell:

```bash
source .env
```

The required variables are:

- `WA_SHOPPING`
- `WA_SHOPPING_ADMIN`
- `WA_REDDIT`
- `WA_GITLAB`
- `WA_WIKIPEDIA`
- `WA_MAP`
- `WA_HOMEPAGE`

Optional:

- `WA_FULL_RESET`

## Task 1 Verification
Run the full Task 1 environment check:

```bash
python scripts/check_env.py
```

The script checks, in order:

1. Python 3.12 and the active interpreter path
2. imports for `torch`, `transformers`, `browsergym`, `webarena`, `playwright`, `optimum`, and `gptqmodel`
3. the local GPTQ model path and `config.json`
4. required `WA_*` variables
5. HTTP reachability of each WebArena URL
6. Playwright Chromium launch
7. `gym.make(f"browsergym/webarena.{TASK_ID}")`
8. one `reset()` call with `headless=True`

If any step fails, the script prints the exact failing component and exits non-zero.

## Model Smoke Test
Once Task 1 is green, you can run the narrow model smoke test:

```bash
python scripts/test_model.py
```

This script loads the local GPTQ model, runs one short generation, and prints the decoded text.

## Troubleshooting
Missing Playwright browsers:

```bash
playwright install chromium
```

Bad WebArena URLs:
- make sure each `WA_*` URL points to the current host
- if your EC2 instance does not have an Elastic IP, stop/start may change the public DNS name

Failed BrowserGym reset:
- verify every required `WA_*` variable is exported in the same shell where you run `check_env.py`
- confirm the WebArena sites are reachable directly in a browser first
- if the host is remote, restart the services on the host before retrying
