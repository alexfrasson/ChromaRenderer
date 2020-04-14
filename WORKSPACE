workspace(name = "chromarenderer")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

new_local_repository(
    name = "cuda",
    build_file = "//third_party/cuda:BUILD.bazel",
    #path = "/usr/local/cuda/",
    path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/",
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
