CudaToolchainInfo = provider(
    doc = "Information about how to invoke the cuda compiler.",
    # In the real world, compiler_path and system_lib might hold File objects,
    # but for simplicity we'll make them strings instead. arch_flags is a list
    # of strings.
    fields = [
        "nvcc_path",
        #"system_lib",
        #"arch_flags"
    ],
)

def _cuda_toolchain_impl(ctx):
    toolchain_info = platform_common.ToolchainInfo(
        cudainfo = CudaToolchainInfo(
            nvcc_path = ctx.attr.nvcc_path,
            #system_lib = ctx.attr.system_lib,
            #arch_flags = ctx.attr.arch_flags,
        ),
    )
    return [toolchain_info]

cuda_toolchain = rule(
    attrs = {
        "nvcc_path": attr.string(),
        #"system_lib": attr.string(),
        #"arch_flags": attr.string_list(),
    },
    implementation = _cuda_toolchain_impl,
)
