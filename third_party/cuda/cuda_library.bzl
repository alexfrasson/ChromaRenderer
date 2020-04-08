load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("@rules_cc//cc:action_names.bzl", "C_COMPILE_ACTION_NAME")

def _get_headers(deps):
    headers = []
    for dep in deps:
        if dep == CcInfo:
            headers.append(dep[CcInfo].compilation_context.headers)
    return headers

# def _filter_files(files, extensions):
#     filtered = []
#     for file in files:
#         if file endswith extensions
#             filtered.append(file)
#     return filtered

def _cuda_library_impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)

    cuda_srcs = ctx.files.srcs
    cuda_hdrs = ctx.files.hdrs

    deps_headers = _get_headers(ctx.attr.deps)

    # get system includes
    # get cuda includes

    compilation_context = cc_common.create_compilation_context()
    linking_context = cc_common.create_linking_context(libraries_to_link = [], user_link_flags = [])

    objt_files = []
    for src in cuda_srcs:
        obj_file = ctx.actions.declare_file("_objs/{}/{}.obj".format(ctx.label.name, src.basename))

        args = ctx.actions.args()

        #args.add("--version")

        feature_configuration = cc_common.configure_features(
            ctx = ctx,
            cc_toolchain = cc_toolchain,
            requested_features = ctx.features,
            unsupported_features = ctx.disabled_features,
        )

        c_compiler_path = cc_common.get_tool_for_action(
            feature_configuration = feature_configuration,
            action_name = C_COMPILE_ACTION_NAME,
        )

        #print(cc_toolchain.compiler)
        #print(c_compiler_path)

        #args.add("--compiler-bindir=\"C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\BuildTools\\VC\\Tools\\MSVC\\14.16.27023\\bin\\Hostx64\\x64\"")
        #args.add("--compiler-bindir={}".format(c_compiler_path))

        args.add("-gencode=arch=compute_30,code=\"sm_30,compute_30\"")
        args.add("-gencode=arch=compute_75,code=\"sm_75,compute_75\"")

        #args.add("--use-local-env")

        args.add("-ccbin", c_compiler_path)

        args.add("-x", "cu")

        args.add("-I\"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\include\"")
        args.add("-I\"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\include\"")
        args.add("-Ithird_party\\cuda-helpers\\include")
        args.add("-Ithird_party\\cuda-helpers\\include\\cuda-helpers")
        args.add("-IChromaRendererCore/src")

        args.add("--keep-dir", obj_file.dirname)

        args.add("-maxrregcount=0")
        args.add("--machine", "64")

        args.add("--compile")

        args.add("-cudart", "static")

        #args.add("-lineinfo")
        #args.add("-G")
        #args.add("-g")

        args.add("-DWIN32")
        args.add("-DWIN64")
        args.add("-DNDEBUG")
        args.add("-D_CONSOLE")
        args.add("-D_MBCS")

        #args.add("-Xcompiler", "\"/EHsc /W1 /nologo /O2 /Fdx64\\Release\\ChromaRendererCore.pdb /FS /Zi  /MD \"")
        args.add("-Xcompiler", "\"/EHsc /W1 /nologo /O2 /FS /Zi /MD\"")

        # args.add_all(deps_headers, before_each = "-I")
        # #args.add_all(cuda_hdrs, before_each = "-I")

        args.add("-o", obj_file)
        args.add(src.path)

        ctx.actions.run(
            progress_message = "Compiling cuda source file {}".format(obj_file.basename),
            #executable = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\bin\\nvcc.exe",
            #executable = "nvcc",
            executable = ctx.executable.nvcc,
            inputs = depset(items = [src] + cuda_hdrs, transitive = deps_headers),
            outputs = [obj_file],
            arguments = [args],
            use_default_shell_env = True,
        )

        objt_files.append(obj_file)

    # link?

    return [
        DefaultInfo(files = depset(objt_files)),
        CcInfo(compilation_context = compilation_context, linking_context = linking_context),
    ]

cuda_library = rule(
    implementation = _cuda_library_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = [".cu"],
        ),
        "hdrs": attr.label_list(
            allow_files = [".h", ".hpp"],
        ),
        "deps": attr.label_list(
            allow_files = True,
            providers = [CcInfo],
        ),
        "_cc_toolchain": attr.label(
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
        ),
        "nvcc": attr.label(
            executable = True,
            cfg = "host",
            allow_files = True,
            default = Label("@cuda//:nvcc"),
        ),
    },
    provides = [CcInfo],
    fragments = ["cpp"],
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
)