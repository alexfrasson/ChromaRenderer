import subprocess

result = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, shell=True, check=True, text=True)

repo_root_folder = result.stdout

subprocess.run(["buildifier", "-r", "-v", repo_root_folder], capture_output=True, shell=True, check=True, text=True)