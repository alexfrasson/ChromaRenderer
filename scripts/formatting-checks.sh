#!/bin/bash

set -eu

function install_buildifier() {
    sudo curl --location --fail "https://github.com/bazelbuild/buildtools/releases/download/3.3.0/buildifier" --output /usr/bin/buildifier
    sudo chmod +x /usr/bin/buildifier
}

function install_clang_format() {
    sudo apt-get -y -qq install clang-format-12 >/dev/null
}

function install_black() {
    pip install black
}

echo
echo "Installing dependencies..."

install_buildifier
install_clang_format
install_black

echo
echo "Running formatting checks..."

python3 ./scripts/format.py --check
