"""Custom rule that uses nvcc to compile cuda source files."""

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "CPP_COMPILE_ACTION_NAME")

def _get_deps_headers(deps):
    headers = []
    for dep in deps:
        if CcInfo in dep:
            headers.append(dep[CcInfo].compilation_context.headers)
    return headers

def _get_deps_includes(deps):
    includes = []
    for dep in deps:
        if CcInfo in dep:
            includes.append(dep[CcInfo].compilation_context.includes)
    return includes

def _get_deps_system_includes(deps):
    system_includes = []
    for dep in deps:
        if CcInfo in dep:
            system_includes.append(dep[CcInfo].compilation_context.system_includes)
    return system_includes

def _get_toolchain_compiler_flags(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    compile_variables = cc_common.create_compile_variables(
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
    )
    flags = [] + cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = CPP_COMPILE_ACTION_NAME,
        variables = compile_variables,
    )
    return flags

def _filter_gcc_flags(flags):
    flags.remove("-fno-canonical-system-headers")
    flags.remove("-Wno-free-nonheap-object")
    flags.remove("-Wunused-but-set-parameter")

def _get_compiler_flags(ctx):
    flags = _get_toolchain_compiler_flags(ctx)
    flags.extend(ctx.fragments.cpp.copts + ctx.fragments.cpp.cxxopts)
    _filter_gcc_flags(flags)
    return flags

def _print_aspect_impl(target, ctx):
    if "cc_" not in ctx.rule.kind:
        return []

    deps_headers = []
    deps_includes = []
    deps_system_includes = []
    rule_srcs = []
    rule_hdrs = []

    if hasattr(ctx.rule.attr, "deps"):
        deps_headers = _get_deps_headers(ctx.rule.attr.deps)
        deps_includes = _get_deps_includes(ctx.rule.attr.deps)
        deps_system_includes = _get_deps_system_includes(ctx.rule.attr.deps)

    if hasattr(ctx.rule.attr, "srcs"):
        for src in ctx.rule.attr.srcs:
            for f in src.files.to_list():
                if "cpp" in f.extension:
                    rule_srcs.append(f)

    if hasattr(ctx.rule.attr, "hdrs"):
        rule_hdrs = [f for t in ctx.rule.attr.hdrs for f in t.files.to_list()]

    flags = _get_compiler_flags(ctx)

    log_files = []
    for src in rule_srcs + rule_hdrs:
        args = ctx.actions.args()
        args.add(src.path)
        args.add("--quiet")
        args.add("--")
        args.add("-I", ".")

        for includes in deps_includes:
            args.add_all(includes, before_each = "-I")

        for system_include in deps_system_includes:
            args.add_all(system_include, before_each = "-isystem")

        args.add_all(flags)

        log_file = ctx.actions.declare_file("{}.clangtidy.log".format(src.basename))
        log_files.append(log_file)

        command = "set -o pipefail\n{} \"$@\" |& tee {}".format(ctx.file._clangtidy.path, log_file.path)

        ctx.actions.run_shell(
            progress_message = "Running clang-tidy on {}".format(src.basename),
            command = command,
            inputs = depset(items = [src] + rule_hdrs, transitive = deps_headers),
            outputs = [log_file],
            arguments = [args],
            use_default_shell_env = True,
        )

    return [
        DefaultInfo(files = depset(log_files)),
        OutputGroupInfo(report = depset(log_files)),
    ]

print_aspect = aspect(
    implementation = _print_aspect_impl,
    attr_aspects = [],
    attrs = {
        "_cc_toolchain": attr.label(
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
        ),
        "_clangtidy": attr.label(
            executable = True,
            cfg = "host",
            allow_single_file = True,
            default = Label("@clang_tidy//:clang-tidy"),
        ),
    },
    fragments = ["cpp"],
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
)
