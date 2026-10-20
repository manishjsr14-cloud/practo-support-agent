"""Run the simple Part 1 commands sequentially and save their real output."""
# What: runs the whole Part 1 pipeline in order (dataset, KB check, index, sample,
#   calibration, evaluation, stress, demo, tests) and saves each step's output.
# Task: Part 1 (all tasks). Used by: run by hand; no other file imports it.
# It runs dataset.py, rag.py, demo.py, evaluate.py and the tests as separate commands.

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


# Save combined process output even when a command fails, then stop on failure.
# Takes: the output filename and the command arguments. Returns: nothing; exits on failure.
# Why: every saved file must be the real output of a real run, including failures.
def run_step(filename, arguments):
    print("Running: " + " ".join([sys.executable] + arguments), flush=True)
    # Use the same Python interpreter (sys.executable) that is running this script, and
    # merge error messages (stderr) into normal output (stdout) so both are saved.
    result = subprocess.run(
        [sys.executable] + arguments, cwd=ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )
    # Save first, then echo to the screen, so the file exists even if we stop below.
    (OUTPUTS / filename).write_text(result.stdout, encoding="utf-8")
    print(result.stdout, end="", flush=True)
    # A nonzero return code means the step failed: stop the pipeline with that code.
    if result.returncode:
        print(f"Stopped: exit code {result.returncode}; output saved to outputs/{filename}")
        raise SystemExit(result.returncode)


# Recreate current logs and run validation, indexing, measurements, demo, and tests.
# Takes: nothing. Returns: nothing.
# Why: one command reproduces every Part 1 output file from scratch.
def main():
    # Each step: (output file in outputs/, arguments passed to Python). Order matters:
    # the index must exist before searching, and calibration before stress/demo/tests.
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
    # Empty every output file first, so a stopped run cannot leave older results behind.
    for filename, arguments in steps:
        (OUTPUTS / filename).write_text("", encoding="utf-8")
    for filename, arguments in steps:
        run_step(filename, arguments)
    print("Part 1 commands completed. Current results are saved in outputs/.")


if __name__ == "__main__":
    main()
