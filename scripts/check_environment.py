"""Print installation checks without requiring TensorFlow."""
import importlib.util
import sys
from handwrite_ai.config import ROOT, ARTIFACTS
from handwrite_ai.demo import train_demo


def main():
    print("HandWrite AI installation check")
    print("Python:", sys.version.split()[0])
    print("Project root:", ROOT)
    for package in ("numpy", "sklearn", "PIL", "matplotlib", "pandas", "streamlit", "tensorflow"):
        print(f"{package:12s}", "installed" if importlib.util.find_spec(package) else "NOT installed")
    _, accuracy, _, _ = train_demo()
    print(f"Working offline demo test accuracy: {accuracy:.4f}")
    print("Saved experiment records:", len(list(ARTIFACTS.glob("*.json"))))


if __name__ == "__main__":
    main()
