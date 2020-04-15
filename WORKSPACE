workspace(name = "chromarenderer")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("//third_party/cuda:cuda_configure.bzl", "cuda_configure")

cuda_configure(name = "cuda")

register_toolchains(
    "//third_party/cuda/toolchain:cuda_linux_toolchain",
    "//third_party/cuda/toolchain:cuda_windows_toolchain",
    # Target patterns are also permitted, so we could have also written:
    # "//bar_tools:all",
)

new_local_repository(
    name = "gtk",
    build_file = "//third_party/gtk:BUILD.bazel",
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
