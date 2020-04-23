load("@rules_cc//cc:defs.bzl", "cc_library")

alias(
    name = "glfw",
    actual = select({
        ":on_linux": ":glfw_x11",
        ":on_windows": ":glfw_win32",
    }),
    visibility = ["//visibility:public"],
)

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

########################################################################################################################
# common
########################################################################################################################

cc_library(
    name = "public_headers",
    hdrs = [
        "include/GLFW/glfw3.h",
        "include/GLFW/glfw3native.h",
    ],
    includes = ["include"],
)

COMMON_HDRS = [
    "src/internal.h",
    "src/mappings.h",
    "src/egl_context.h",
    "src/osmesa_context.h",
]

COMMON_SRCS = [
    "src/context.c",
    "src/init.c",
    "src/input.c",
    "src/monitor.c",
    "src/vulkan.c",
    "src/window.c",
    "src/egl_context.c",
    "src/osmesa_context.c",
]

########################################################################################################################
# x11
########################################################################################################################

cc_library(
    name = "glfw_x11",
    srcs = [
        "src/glx_context.c",
        "src/posix_thread.c",
        "src/posix_time.c",
        "src/x11_init.c",
        "src/x11_monitor.c",
        "src/x11_window.c",
        "src/xkb_unicode.c",
        "src/linux_joystick.c",
    ] + COMMON_SRCS,
    hdrs = [
        "src/glx_context.h",
        "src/posix_thread.h",
        "src/posix_time.h",
        "src/x11_platform.h",
        "src/xkb_unicode.h",
        "src/linux_joystick.h",
    ] + COMMON_HDRS,
    defines = ["_GLFW_X11"],
    linkopts = ["-lX11 -ldl"],
    deps = [":public_headers"],
)

########################################################################################################################
# win32
########################################################################################################################

cc_library(
    name = "glfw_win32",
    srcs = [
        "src/wgl_context.c",
        "src/win32_init.c",
        "src/win32_joystick.c",
        "src/win32_monitor.c",
        "src/win32_thread.c",
        "src/win32_time.c",
        "src/win32_window.c",
    ] + COMMON_SRCS,
    hdrs = [
        "src/wgl_context.h",
        "src/win32_joystick.h",
        "src/win32_platform.h",
    ] + COMMON_HDRS,
    defines = ["_GLFW_WIN32"],
    linkopts = [
        "-DEFAULTLIB:gdi32.lib",
        "-DEFAULTLIB:ole32.lib",
        "-DEFAULTLIB:propsys.lib",
        "-DEFAULTLIB:shlwapi.lib",
    ],
    deps = [":public_headers"],
)
