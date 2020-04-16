@echo off

echo Current directory: %cd%

echo Installing CUDA toolkit 10.0
%TEMP%\cuda_10.0.130_411.31_win10.exe -s nvcc_10.0 ^
                                         curand_10.0 ^
                                         curand_dev_10.0 ^
                                         cudart_10.0

if NOT EXIST "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin\cudart64_100.dll" ( 
    echo "Failed to install CUDA"
    exit /B 1
)

nvcc -V

bazel build //ChromaRenderer:chroma-renderer