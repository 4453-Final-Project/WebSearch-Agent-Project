from importlib.metadata import packages_distributions
import json
import sys
from pathlib import Path

def checkVersion():
    # Compares current python version with the needed 3.12.X.
    version_check = (3, 12)
    if sys.version_info[:2] != version_check:
        raise RuntimeError(f"Python{version_check[0]}.{version_check[1]} version is required.")

def checkPackages():
    # Checks if all the packages have been properly installed.
    # If dependencies change simply add or remove variables from packages.
    packages = [
        "torch",
        "transformers",
        "browsergym",
        "playwright",
        "optimum",
        "gptqmodel",
        "webarena",
    ]
    failed = False

    installed = packages_distributions()
    allPackages = list(installed.keys())
    for package in packages:
        if package not in allPackages:
            print(f"You do not have the {package} module")
            failed = True
    if failed:
        raise RuntimeError(f"Pacakges missing, please install the required packages before continuing.")

def checkModel():
    # First check if the file format is as expected
    SCRIPT_DIR = Path(__file__).resolve().parent
    MODEL_PATH = (SCRIPT_DIR / ".." / ".." / "models" / "Qwen2.5-3B-Instruct-GPTQ-Int4").resolve()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Local model directory not found: {MODEL_PATH}\n"
            "Download it with: hf download Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 "
            f"--local-dir {MODEL_PATH}"
        )
    # Next check if config.json has initialized correctly and the quant type is gptq.
    with open(f"{MODEL_PATH}/config.json") as f:
        data = json.load(f)
        quant_config = data.get("quantization_config")
        if quant_config is None:
            raise RuntimeError(f"Quant config not found")
        quant_type = quant_config.get("quant_method", "").lower()
        if quant_type != "gptq":
            raise RuntimeError(f"ERROR: Expected gptq but got '{quant_type}'")
def main():
    checkVersion()
    checkPackages()
    checkModel()
    print("Verification complete!")
main()
