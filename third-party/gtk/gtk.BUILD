load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "gtk",
    srcs = ["lib/x86_64-linux-gnu/libgtk-3.so"],
    hdrs = glob(["include/gtk-3.0/**/*.h"]) + ["include/limits.h"],
    includes = ["include/gtk-3.0"],
    deps = [
        ":atk",
        ":cairo",
        ":gdk-pixbuf",
        ":glib",
        ":pango",
    ],
)

cc_library(
    name = "atk",
    hdrs = glob(["include/atk-1.0/**/*.h"]),
    includes = ["include/atk-1.0"],
)

cc_library(
    name = "cairo",
    hdrs = glob(["include/cairo/**/*.h"]),
    includes = ["include/cairo"],
)

cc_library(
    name = "glib",
    srcs = [
        "lib/x86_64-linux-gnu/libglib-2.0.so",
        "lib/x86_64-linux-gnu/libgobject-2.0.so",
    ],
    hdrs = glob([
        "lib/x86_64-linux-gnu/glib-2.0/**/*.h",
        "include/glib-2.0/**/*.h",
    ]),
    includes = [
        "include/glib-2.0",
        "lib/x86_64-linux-gnu/glib-2.0/include",
    ],
)

cc_library(
    name = "gdk-pixbuf",
    hdrs = glob(["include/gdk-pixbuf-2.0/**/*.h"]),
    includes = ["include/gdk-pixbuf-2.0"],
)

cc_library(
    name = "pango",
    hdrs = glob(["include/pango-1.0/**/*.h"]),
    includes = ["include/pango-1.0"],
)
