"""Run the simple Part 1 commands sequentially and save their real output."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


# Save combined process output even when a command fails, then stop on failure.
def run_step(filename, arguments):
    print("Running: " + " ".join([sys.executable] + arguments), flush=True)
    result = subprocess.run(
        [sys.executable] + arguments, cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )
    (OUTPUTS / filename).write_text(result.stdout, encoding="utf-8")
    print(result.stdout, end="", flush=True)
    if result.returncode:
        print(f"Stopped: exit code {result.returncode}; output saved to outputs/{filename}")
        raise SystemExit(result.returncode)


# Recreate current logs and run validation, indexing, measurements, demo, and tests.
def main():
    steps = [
        ("dataset.txt", ["dataset.py"]),
        ("kb.txt", ["rag.py", "validate"]),
        ("index.txt", ["rag.py", "index"]),
        ("sample_retrieval.txt", ["demo.py", "--sample"]),
        ("calibration.txt", ["evaluate.py", "calibration"]),
        ("evaluation.txt", ["evaluate.py", "evaluation"]),
        ("near_topic.txt", ["evaluate.py", "stress"]),
        ("demo.txt", ["demo.py"]),
        ("tests.txt", ["-m", "unittest", "discover", "-s", "tests", "-v"]),
    ]
    OUTPUTS.mkdir(exist_ok=True)
    for filename, arguments in steps:
        (OUTPUTS / filename).write_text("", encoding="utf-8")
    for filename, arguments in steps:
        run_step(filename, arguments)
    print("Part 1 commands completed. Current results are saved in outputs/.")


if __name__ == "__main__":
    main()
