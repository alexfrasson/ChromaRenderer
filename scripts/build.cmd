@echo off

echo "Current directory: %cd%"
echo "TEMP: %TEMP%"

set CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v10.2
echo "CUDA_PATH: %CUDA_PATH%"

echo "Downloading CUDA toolkit 10.2"
curl -L --url "http://developer.download.nvidia.com/compute/cuda/10.2/Prod/network_installers/cuda_10.2.89_win10_network.exe" --output "%TEMP%\cuda_installer.exe"
echo "Installing CUDA toolkit 10.2"
%TEMP%\cuda_installer.exe -s nvcc_10.2 ^
                             curand_10.2 ^
                             curand_dev_10.2 ^
                             cudart_10.2

if NOT EXIST "%CUDA_PATH%\bin\cudart64_102.dll" ( 
    echo "CUDA installation failed!"
    exit /B 1
)

set PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%
nvcc -V

echo "Downloading bazelisk 1.3.0"
set BAZELISK_PATH=%TEMP%\bazelisk
set PATH=%BAZELISK_PATH%;%PATH%

mkdir %BAZELISK_PATH%

curl -L --url "https://github.com/bazelbuild/bazelisk/releases/download/v1.3.0/bazelisk-windows-amd64.exe" --output "%BAZELISK_PATH%\bazel.exe"

if NOT EXIST "%BAZELISK_PATH%\bazel.exe" ( 
    echo "Bazelisk installation failed!"
    exit /B 1
)

set CC_CONFIGURE_DEBUG=1

bazel build --config=windows --verbose_failures //...
bazel test --config=windows --verbose_failures //...
