from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension
from torch import cuda

ext_module_cpp = cpp_extension.CppExtension(
    'ops_cc', 
    ['ops/ops.cc'],
    extra_compile_args=['-fopenmp', '-Wall', '-Werror']
)
ext_modules = [ext_module_cpp]

"""
if cuda.is_available():
    ext_module_cuda = cpp_extension.CUDAExtension(
        "ops_cuda",
        ["ops/ops.cu"]
    )
    ext_modules.append(ext_module_cuda)
"""

setup(
    name='ops',
    packages = find_packages(),
    ext_modules = ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
