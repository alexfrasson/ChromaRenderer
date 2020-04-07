@echo off

echo Current directory: %cd%

echo Downloading CUDA toolkit 10.0
appveyor DownloadFile https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_411.31_win10 -FileName cuda_10.0.130_411.31_win10.exe

echo Installing CUDA toolkit 10.0
cuda_10.0.130_411.31_win10.exe -s nvcc_10.0 ^
                                  curand_10.0 ^
                                  curand_dev_10.0 ^
                                  cudart_10.0

::dir "%ProgramFiles%"
::dir "C:\Program Files"
::dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
::dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0"
::dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin"

if NOT EXIST "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin\cudart64_100.dll" ( 
echo "Failed to install CUDA"
exit /B 1
)

set PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin;%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp;%PATH%

nvcc -V

echo Downloading bazelisk
appveyor DownloadFile https://github.com/bazelbuild/bazelisk/releases/download/v1.3.0/bazelisk-windows-amd64.exe -FileName bazelisk.exe

cd "C:\projects\chromarenderer\ChromaRenderer"

"C:\projects\chromarenderer\bazelisk.exe" build //ChromaRenderer:chroma-renderer

::"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"

::msbuild "C:\projects\chromarenderer\ChromaRenderer\ChromaRenderer.sln" /verbosity:detailed /logger:"C:\Program Files\AppVeyor\BuildAgent\Appveyor.MSBuildLogger.dll"
