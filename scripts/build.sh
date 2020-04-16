#!/bin/bash

function install_cuda()
{
    echo "Downloading CUDA installer..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub > /dev/null
    sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" > /dev/null
    sudo apt-get -qq update > /dev/null

    echo "Installing CUDA..."
    sudo apt-get -y -qq install cuda-nvcc-10-2 \
                                cuda-curand-dev-10-2 \
                                cuda-cudart-dev-10-2 \
                                > /dev/null

    export CUDA_PATH="/usr/local/cuda-10.2"
    echo "CUDA_PATH set to '$CUDA_PATH'"
}

function install_deps()
{
    install_cuda

    echo "Installing other dependencies..."
    sudo apt-get -y -qq install libgtk-3-dev > /dev/null
}

install_deps

echo
echo "Building project..."

export CC_CONFIGURE_DEBUG=1

bazelisk build --config=linux //ChromaRenderer:chroma-renderer --verbose_failures