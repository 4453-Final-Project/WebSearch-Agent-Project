import sys
from pathlib import Path
from time import perf_counter


print("Status: starting test_model.py...", flush=True)

# Resolve the script location up front so every other path in this file can be
# constructed relative to the checked-in repository layout instead of the shell's
# current working directory. This makes the script behave the same whether it is
# launched from the repo root, the scripts directory, or elsewhere.
SCRIPT_DIR = Path(__file__).resolve().parent

# The project layout for this assignment places the repository and the shared
# `.venv`/`models` directories side-by-side under the same parent directory.
# `WORKSPACE_DIR` is that shared parent directory.
WORKSPACE_DIR = SCRIPT_DIR.parent.parent


def add_local_venv_site_packages() -> None:
    print("Status: searching for local virtualenv site-packages...", flush=True)

    # The user may forget to activate the repository's virtual environment before
    # running this script. When that happens, imports such as `transformers` may
    # fail even though the dependency is already installed in the local `.venv`.
    # This helper searches for the most likely site-packages directories inside
    # that virtual environment and prepends them to `sys.path` so the script can
    # recover automatically.
    venv_dir = WORKSPACE_DIR / ".venv"

    # Linux/macOS virtual environments typically store packages under
    # `.venv/lib/pythonX.Y/site-packages`, while Windows uses
    # `.venv/Lib/site-packages`. We look for both layouts so the script remains
    # readable and portable across environments.
    candidate_dirs = sorted(venv_dir.glob("lib/python*/site-packages"))
    candidate_dirs.extend(sorted(venv_dir.glob("Lib/site-packages")))

    for site_packages in candidate_dirs:
        site_packages_str = str(site_packages.resolve())
        # Only prepend directories that actually exist, and avoid duplicating
        # entries already on `sys.path`.
        if site_packages.exists() and site_packages_str not in sys.path:
            print(f"Status: adding site-packages path: {site_packages_str}", flush=True)
            sys.path.insert(0, site_packages_str)


try:
    print("Status: importing AutoModelForCausalLM, AutoTokenizer from transformers...", flush=True)

    # First try the normal import path. This is the fast path when the caller
    # already activated the correct environment.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Status: transformers import complete.", flush=True)
except ModuleNotFoundError as exc:
    # If some nested dependency inside `transformers` is missing, we should not
    # mask that error here. Only handle the case where the top-level
    # `transformers` package itself cannot be found.
    if exc.name != "transformers":
        raise

    print("Status: transformers not found in active environment. Retrying with local .venv...", flush=True)

    # Retry the import after adding packages from the local `.venv`.
    add_local_venv_site_packages()

    try:
        print("Status: importing transformers from local .venv...", flush=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Status: transformers import complete.", flush=True)
    except ModuleNotFoundError as inner_exc:
        # Surface a direct, actionable error message that points to the project
        # virtual environment and the requirements file, rather than leaving the
        # user with a generic import failure.
        raise ModuleNotFoundError(
            "transformers is not installed for this script. "
            f"Expected to find it in {WORKSPACE_DIR / '.venv'} or your active Python environment.\n"
            "Activate the repo virtualenv with: source ../.venv/bin/activate\n"
            "Or install dependencies with: pip install -r requirements.txt"
        ) from inner_exc


# The model is expected to have been downloaded locally outside the git repo,
# under the shared `models/` directory described in the README. Resolving this
# path from `SCRIPT_DIR` ensures the script always targets the same local model
# directory regardless of where the command is run from.
MODEL_PATH = (SCRIPT_DIR / ".." / ".." / "models" / "Qwen2.5-3B-Instruct").resolve()


def main() -> None:
    print("Status: validating local model path...", flush=True)

    # Fail early with a precise path if the model has not been downloaded yet.
    # This is clearer than letting `transformers` error later with a less obvious
    # message.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Local model directory not found: {MODEL_PATH}\n"
            "Download it with: hf download Qwen/Qwen2.5-3B-Instruct "
            f"--local-dir {MODEL_PATH}"
        )

    # Keep the test prompt intentionally simple. The goal of this script is not
    # to evaluate response quality; it is to prove the local tokenizer and model
    # can load and produce at least one completion end-to-end.
    prompt = "Hello"

    # Track total wall-clock time for the entire smoke test so the user can see
    # how long the complete startup + generation path takes on their machine.
    overall_start = perf_counter()

    # Print a readable summary before the slow work starts so a user waiting on a
    # large local model load understands what the script is about to do.
    print("General Specification/Expected output:")
    print("Load the local tokenizer and model, run one short generation, and print the decoded result.")
    print(f"User prompt: {prompt}")
    print(f"Model path: {MODEL_PATH}")
    print("Status: loading tokenizer...", flush=True)

    # Measure tokenizer initialization separately because it is usually fast and
    # gives an early sign that the local files are readable.
    tokenizer_start = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    print(f"Status: tokenizer loaded in {perf_counter() - tokenizer_start:.1f}s", flush=True)

    print("Status: loading model weights...", flush=True)

    # Loading the model weights is typically the slowest step in this script. We
    # request local-only loading because this test is meant to validate the local
    # download, not fetch anything from Hugging Face at runtime.
    model_start = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype="auto",
        device_map="auto",
    )
    print(f"Status: model loaded in {perf_counter() - model_start:.1f}s", flush=True)

    print("Status: tokenizing prompt...", flush=True)

    # Convert the human-readable prompt into tensors and move them onto the same
    # device placement chosen for the model so generation can run without device
    # mismatch errors.
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Status: generating output...", flush=True)

    # Run a very short generation. Limiting `max_new_tokens` keeps this a quick
    # smoke test rather than a full benchmark or evaluation script.
    generation_start = perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(f"Status: generation finished in {perf_counter() - generation_start:.1f}s", flush=True)

    # Decode the generated token IDs back into normal text so the caller can see
    # the model's response directly in the terminal.
    model_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Model Output: {model_output}")
    print(f"Total time: {perf_counter() - overall_start:.1f}s")


# Keep the script import-safe: importing this module for inspection should not
# immediately load a multi-gigabyte model. The smoke test only runs when the file
# is executed as a script.
if __name__ == "__main__":
    main()
