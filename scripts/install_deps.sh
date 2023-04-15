#!/bin/bash

set -eu

function install_cuda() {
    echo "Downloading CUDA installer..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    
    sudo apt-get -qq update >/dev/null

    echo "Installing CUDA..."
    sudo apt-get -y -qq install cuda-nvcc-12-1 \
        libcurand-dev-12-1 \
        cuda-cudart-dev-12-1 \
        cuda-driver-dev-12-1 \
        >/dev/null
}

function install_deps() {
    install_cuda

    echo "Installing other dependencies..."
    sudo apt-get -y -qq install libgtk-3-dev \
        clang-tidy-12 \
        clang-12 \
        gcc-10 \
        g++-10 \
        >/dev/null
}

install_deps
