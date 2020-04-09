import subprocess
import glob

# Bazel's buildifier

result = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"],
    capture_output=True,
    shell=True,
    check=True,
    text=True,
)

repo_root_folder = result.stdout

subprocess.run(
    ["buildifier", "-r", "-v", repo_root_folder],
    capture_output=True,
    shell=True,
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
        shell=True,
        check=True,
        text=True,
    )
