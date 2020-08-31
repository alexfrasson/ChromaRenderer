load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "on_linux",
    constraint_values = [
        "@platforms//os:linux",
    ],
)

config_setting(
    name = "on_windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

cc_library(
    name = "cuda_headers",
    hdrs = glob(
        include = [
            "cuda/include/**/*.h",
            "cuda/include/**/*.hpp",
        ],
        allow_empty = False,
    ),
    includes = ["cuda/include"],
)

cc_library(
    name = "cuda",
    srcs = select({
        ":on_linux": [],
        ":on_windows": ["cuda/lib/x64/cuda.lib"],
    }),
    #linkopts = ["-ldl"],
)

cc_library(
    name = "cuda_runtime",
    srcs = select({
        ":on_linux": ["cuda/lib64/libcudart_static.a"],
        ":on_windows": ["cuda/lib/x64/cudart_static.lib"],
    }),
    linkopts = select({
        "//:on_linux": [
            "-lpthread",
            "-lrt",
        ],
        "//:on_windows": [],
    }),
    deps = [":cuda"],
)

cc_library(
    name = "curand_static",
    srcs = ["cuda/lib/x64/curand.lib"],
    deps = [
        #":culibos",
    ],
)

filegroup(
    name = "nvcc",
    srcs = select({
        ":on_linux": ["cuda/bin/nvcc"],
        ":on_windows": ["cuda/bin/nvcc.exe"],
    }),
)
