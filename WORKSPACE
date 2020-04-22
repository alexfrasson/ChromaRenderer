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
    build_file = "//third-party/gtk:BUILD.bazel",
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
    build_file = "//third-party/assimp:BUILD.bazel",
    sha256 = "11310ec1f2ad2cd46b95ba88faca8f7aaa1efe9aa12605c55e3de2b977b3dbfc",
    strip_prefix = "assimp-5.0.1",
    urls = ["https://github.com/assimp/assimp/archive/v5.0.1.tar.gz"],
    workspace_file = "//third-party/assimp:WORKSPACE",
)
