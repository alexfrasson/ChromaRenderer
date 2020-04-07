@echo off

echo Current directory: %cd%

echo Downloading CUDA toolkit 10.0
appveyor DownloadFile https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_411.31_win10 -FileName %temp%\cuda_10.0.130_411.31_win10.exe

echo Installing CUDA toolkit 10.0
%temp%\cuda_10.0.130_411.31_win10.exe -s nvcc_10.0 ^
                                         curand_10.0 ^
                                         curand_dev_10.0 ^
                                         cudart_10.0

if NOT EXIST "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin\cudart64_100.dll" ( 
    echo "Failed to install CUDA"
    exit /B 1
)

set PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin;%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp;%PATH%

nvcc -V

echo Downloading bazelisk 1.3.0
mkdir %temp%\bazelisk
appveyor DownloadFile https://github.com/bazelbuild/bazelisk/releases/download/v1.3.0/bazelisk-windows-amd64.exe -FileName %temp%\bazelisk\bazel.exe

set PATH=%temp%\bazelisk;%PATH%

bazel build //ChromaRenderer:chroma-renderer