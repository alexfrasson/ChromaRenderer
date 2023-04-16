# Chroma Renderer  ![CI](https://github.com/alexfrasson/ChromaRenderer/workflows/CI/badge.svg?branch=master)

Yet another path tracer. It was written during my time in college and it is slowly being refactored. Far from a good example of clean code, design patterns or C++ development in general.

![alt text](chroma-renderer/samples/sample.jpg?raw=true)

## Required system installed dependencies
- [Bazel 3.0](https://docs.bazel.build/versions/master/install.html) or [Bazelisk](https://github.com/bazelbuild/bazelisk/releases)
- Cuda 12.1 (older releases may also work)
  - Environment variable **CUDA_PATH** must be set. Examples:
    - Windows: `%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.1`
    - Ubuntu: `/usr/local/cuda-12.1`
    
On Windows:
- MSVC++ 14.1 or later
- See [Build C++ with MSVC](https://docs.bazel.build/versions/master/windows.html#build-c-with-msvc) in case you have multiple MSVC++ versions installed on your system

On Ubuntu:
- g++-10 or clang-12
- libgtk-3-dev

## Building

Simply run the following command from within the workspace. It should fetch all dependencies, build, and run the application. Choose the correct configuration based on your OS/compiler.

`bazel run --config=[msvc|gcc|clang] //chroma-renderer/gui`

An example scene and several unclipped environment maps can be found in `chroma-renderer/resources`.
