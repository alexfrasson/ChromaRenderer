#!/usr/bin/python3
import subprocess
import glob

# Bazel's buildifier
result = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, check=True, text=True,
)

repo_root_folder = result.stdout.rstrip("\n")

print("Running buildifier on {}".format(repo_root_folder))

subprocess.run(
    ["buildifier", "-r", "-v", repo_root_folder],
    capture_output=True,
    check=True,
    text=True,
)

# Clang format
extensions = [".cu", ".glsl", ".cpp", ".h"]
folders = ["ChromaRenderer", "ChromaRendererCore"]

files = []

for extension in extensions:
    for folder in folders:
        files = files + glob.glob("{}/**/*{}".format(folder, extension), recursive=True)

for filename in files:
    print("Running clang-format on {}".format(filename))
    subprocess.run(
        [
            "clang-format",
            "-i",
            "--verbose",
            "--style=file",
            "--fallback-style=none",
            filename,
        ],
        capture_output=True,
        check=True,
        text=True,
    )

# Black for python files
print("Running black on './scripts'")

subprocess.run(
    ["python3", "-m", "black", "./scripts",],
    capture_output=True,
    check=True,
    text=True,
)
