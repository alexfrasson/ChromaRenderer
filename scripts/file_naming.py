#!/usr/bin/python3.8
import re
import os
import sys
import glob
import argparse

camel_to_snake_regex = re.compile("((?<=[a-z0-9])[A-Z]|(?!^)(?<!_)[A-Z](?=[a-z]))")


def camel_to_snake(name):
    name = name.replace("-", "_")
    return camel_to_snake_regex.sub(r"_\1", name).lower()


def get_files():
    extensions = [".cu", ".glsl", ".cpp", ".h"]
    folders = ["chroma-renderer"]

    files = []
    for extension in extensions:
        for folder in folders:
            files = files + glob.glob(
                "{}/**/*{}".format(folder, extension), recursive=True
            )
    return files


def get_files_to_be_fixed(files):
    files_to_be_fixed = []
    for filename in files:
        name, extension = os.path.splitext(os.path.basename(filename))
        snake_name = camel_to_snake(name)

        if snake_name != name:
            dirname = os.path.dirname(filename)
            correct_filename = os.path.join(dirname, snake_name + extension)
            files_to_be_fixed.append((filename, correct_filename))

    return files_to_be_fixed


def replace_name_in_files(files, original_name, new_name):
    for file in files:
        with open(file) as f:
            data = f.read()
        with open(file, "w") as f:
            data = data.replace(original_name, new_name)
            f.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--fix",
        action="store_true",
        help="Rename files to snake_case.",
    )
    args = parser.parse_args()

    print("Checking file names...")

    files = get_files()
    files_to_be_fixed = get_files_to_be_fixed(files)

    if args.fix:
        for original_name, correct_name in files_to_be_fixed:
            print("Replacing occurences of '{}'".format(original_name))
            replace_name_in_files(files, original_name, correct_name)

    for original_name, correct_name in files_to_be_fixed:
        if args.fix:
            os.rename(original_name, correct_name)
            print("'{}' renamed to '{}'".format(original_name, correct_name))
        else:
            print("'{}' should be '{}'".format(original_name, correct_name))

    violations_found = len(files_to_be_fixed) != 0

    if args.fix:
        return True
    elif not violations_found:
        print("OK")

    return violations_found


if __name__ == "__main__":
    sys.exit(main())
