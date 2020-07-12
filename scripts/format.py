#!/usr/bin/python3.7
import subprocess
import glob
import argparse
import sys


def handle_bazel_files(check_only):
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        check=True,
        text=True,
    )

    repo_root_folder = result.stdout.rstrip("\n")

    print("Running buildifier on {}".format(repo_root_folder))

    if check_only:
        proc_result = subprocess.run(
            ["buildifier", "-r", "-v", "-mode", "check", repo_root_folder],
            capture_output=True,
            text=True,
        )
        if proc_result.returncode != 0:
            print("Formatting issues found in the following files:")
            print("  " + proc_result.stderr.replace("\n", "\n  "))
            return False
    else:
        subprocess.run(
            ["buildifier", "-r", "-v", repo_root_folder],
            capture_output=True,
            check=True,
            text=True,
        )
    return True


def handle_cpp_files(check_only):
    proc_result = subprocess.run(
        ["clang-format", "--version"], capture_output=True, check=True, text=True
    )
    print(proc_result.stdout)

    extensions = [".cu", ".glsl", ".cpp", ".h"]
    folders = ["chroma-renderer"]

    files_to_format = []

    for extension in extensions:
        for folder in folders:
            files_to_format = files_to_format + glob.glob(
                "{}/**/*{}".format(folder, extension), recursive=True
            )

    files_with_errors = []

    for filename in files_to_format:
        # print("Running clang-format on {}".format(filename))
        if check_only:
            proc_result = subprocess.run(
                [
                    "clang-format",
                    "--verbose",
                    "--style=file",
                    "--fallback-style=none",
                    "--dry-run",
                    "-Werror",
                    filename,
                ],
                capture_output=True,
                text=True,
            )
            if proc_result.returncode != 0:
                print(proc_result.stderr)
                files_with_errors.append(filename)

        else:
            subprocess.run(
                [
                    "clang-format",
                    "--verbose",
                    "--style=file",
                    "--fallback-style=none",
                    "-i",
                    filename,
                ],
                capture_output=True,
                check=True,
                text=True,
            )

    if check_only and files_with_errors:
        print("Formatting issues found in the following files:")
        for file in files_with_errors:
            print("  " + file)
        return False
    return True


def handle_python_files():
    print("Running black on './scripts'")

    subprocess.run(
        ["python3", "-m", "black", "./scripts",],
        capture_output=True,
        check=True,
        text=True,
    )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--check",
        action="store_true",
        help="Check whether code properly formatted. No formatting is applied.",
    )
    args = parser.parse_args()

    status = True

    status &= handle_bazel_files(args.check)
    status &= handle_cpp_files(args.check)
    status &= handle_python_files()

    return not status


if __name__ == "__main__":
    sys.exit(main())
