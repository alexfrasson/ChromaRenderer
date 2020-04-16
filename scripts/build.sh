#!/bin/bash

function install_cuda()
{
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
    sudo apt-get update
    sudo apt-get -y -qq install cuda-nvcc-10-2 \
                                cuda-curand-dev-10-2 \
                                cuda-cudart-dev-10-2

    export CUDA_PATH="/usr/local/cuda-10.2"
}

function install_deps()
{
    install_cuda

    sudo apt-get -y -qq install libgtk-3-dev
}

install_deps

export CC_CONFIGURE_DEBUG=1

bazelisk version
bazelisk build --config=linux //ChromaRenderer:chroma-renderer --verbose_failures