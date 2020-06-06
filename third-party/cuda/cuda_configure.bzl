""" Generate build rule for cuda dependencies"""

def _get_cuda_path(repository_ctx):
    if not "CUDA_PATH" in repository_ctx.os.environ:
        fail("Environment variable 'CUDA_PATH' could not be found.")
    return repository_ctx.os.environ["CUDA_PATH"]

def _impl(repository_ctx):
    cuda_path = _get_cuda_path(repository_ctx)
    repository_ctx.symlink(cuda_path, "cuda")
    repository_ctx.symlink(repository_ctx.attr.build_file, "BUILD.bazel")

cuda_configure = repository_rule(
    implementation = _impl,
    environ = ["CUDA_PATH"],
    attrs = {
        "build_file": attr.label(
            allow_files = True,
            default = Label("//third-party/cuda:cuda.BUILD"),
        ),
    },
)
