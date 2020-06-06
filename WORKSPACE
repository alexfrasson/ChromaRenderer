workspace(name = "chromarenderer")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third-party/cuda:cuda_configure.bzl", "cuda_configure")

cuda_configure(name = "cuda")

register_toolchains(
    "//third-party/cuda/toolchain:cuda_linux_toolchain",
    "//third-party/cuda/toolchain:cuda_windows_toolchain",
    # Target patterns are also permitted, so we could have also written:
    # "//bar_tools:all",
)

new_local_repository(
    name = "gtk",
    build_file = "//third-party/gtk:gtk.BUILD",
    path = "/usr/",
)

git_repository(
    name = "bazel_skylib",
    commit = "e59b620b392a8ebbcf25879fc3fde52b4dc77535",
    remote = "https://github.com/bazelbuild/bazel-skylib.git",
    shallow_since = "1570639401 -0400",
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

http_archive(
    name = "assimp",
    build_file = "//third-party/assimp:assimp.BUILD",
    sha256 = "11310ec1f2ad2cd46b95ba88faca8f7aaa1efe9aa12605c55e3de2b977b3dbfc",
    strip_prefix = "assimp-5.0.1",
    urls = ["https://github.com/assimp/assimp/archive/v5.0.1.tar.gz"],
    workspace_file = "//third-party/assimp:WORKSPACE",
)

http_archive(
    name = "glfw",
    build_file = "//third-party/glfw:glfw.BUILD",
    sha256 = "98768e12e615fbe9f3386f5bbfeb91b5a3b45a8c4c77159cef06b1f6ff749537",
    strip_prefix = "glfw-3.3.2",
    urls = ["https://github.com/glfw/glfw/archive/3.3.2.tar.gz"],
    workspace_file = "//third-party/glfw:WORKSPACE",
)

http_archive(
    name = "googletest",
    sha256 = "9dc9157a9a1551ec7a7e43daea9a694a0bb5fb8bec81235d8a1e6ef64c716dcb",
    strip_prefix = "googletest-release-1.10.0",
    urls = ["https://github.com/google/googletest/archive/release-1.10.0.tar.gz"],
)
