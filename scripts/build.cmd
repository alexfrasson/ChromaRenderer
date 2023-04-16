@echo off

echo "Current directory: %cd%"
echo "TEMP: %TEMP%"

set CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.1
echo "CUDA_PATH: %CUDA_PATH%"

echo "Downloading CUDA toolkit 12.1"
curl -L --url "https://developer.download.nvidia.com/compute/cuda/12.1.0/network_installers/cuda_12.1.0_windows_network.exe" --output "%TEMP%\cuda_installer.exe"
echo "Installing CUDA toolkit 12.1"
%TEMP%\cuda_installer.exe -s nvcc_12.1 ^
                             curand_12.1 ^
                             curand_dev_12.1 ^
                             thrust_12.1 ^
                             cudart_12.1

if NOT EXIST "%CUDA_PATH%\bin\cudart64_12.dll" ( 
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

bazel build --config=msvc --verbose_failures //...
bazel test --config=msvc --verbose_failures //...
