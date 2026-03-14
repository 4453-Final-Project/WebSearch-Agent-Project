import sys
from pathlib import Path
from time import perf_counter


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent.parent


def add_local_venv_site_packages() -> None:
    venv_dir = WORKSPACE_DIR / ".venv"
    candidate_dirs = sorted(venv_dir.glob("lib/python*/site-packages"))
    candidate_dirs.extend(sorted(venv_dir.glob("Lib/site-packages")))

    for site_packages in candidate_dirs:
        site_packages_str = str(site_packages.resolve())
        if site_packages.exists() and site_packages_str not in sys.path:
            sys.path.insert(0, site_packages_str)


try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError as exc:
    if exc.name != "transformers":
        raise

    add_local_venv_site_packages()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as inner_exc:
        raise ModuleNotFoundError(
            "transformers is not installed for this script. "
            f"Expected to find it in {WORKSPACE_DIR / '.venv'} or your active Python environment.\n"
            "Activate the repo virtualenv with: source ../.venv/bin/activate\n"
            "Or install dependencies with: pip install -r requirements.txt"
        ) from inner_exc


MODEL_PATH = (SCRIPT_DIR / ".." / ".." / "models" / "Qwen2.5-3B-Instruct").resolve()


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Local model directory not found: {MODEL_PATH}\n"
            "Download it with: hf download Qwen/Qwen2.5-3B-Instruct "
            f"--local-dir {MODEL_PATH}"
        )

    prompt = "Hello"
    overall_start = perf_counter()

    print("General Specification/Expected output:")
    print("Load the local tokenizer and model, run one short generation, and print the decoded result.")
    print(f"User prompt: {prompt}")
    print(f"Model path: {MODEL_PATH}")
    print("Status: loading tokenizer...")
    tokenizer_start = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    print(f"Status: tokenizer loaded in {perf_counter() - tokenizer_start:.1f}s")

    print("Status: loading model weights...")
    model_start = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype="auto",
        device_map="auto",
    )
    print(f"Status: model loaded in {perf_counter() - model_start:.1f}s")

    print("Status: tokenizing prompt...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Status: generating output...")
    generation_start = perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(f"Status: generation finished in {perf_counter() - generation_start:.1f}s")
    model_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Model Output: {model_output}")
    print(f"Total time: {perf_counter() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
