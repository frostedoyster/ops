from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension
from torch import cuda
from distutils.command.install_lib import install_lib as _install_lib
import os
import re


'''
Hack to remove lib.*.so from the output .so files.
'''

def batch_rename(src, dst, src_dir_fd=None, dst_dir_fd=None):
    '''Same as os.rename, but returns the renaming result.'''
    os.rename(src, dst,
              src_dir_fd=src_dir_fd,
              dst_dir_fd=dst_dir_fd)
    return dst


class _CommandInstallCythonized(_install_lib):
    def __init__(self, *args, **kwargs):
        _install_lib.__init__(self, *args, **kwargs)

    def install(self):
        # let the distutils' install_lib do the hard work
        outfiles = _install_lib.install(self)
        # batch rename the outfiles:
        # for each file, match string between
        # second last and last dot and trim it
        matcher = re.compile('\.([^.]+)\.so$')
        return [batch_rename(file, re.sub(matcher, '.so', file))
                for file in outfiles]


ext_module_cpp = cpp_extension.CppExtension(
    '.ops.ops_cc',
    ['sparse_ops/ops/ops.cc'],
    extra_compile_args=['-fopenmp', '-Wall', '-Werror']
)

ext_modules = [ext_module_cpp]

if cuda.is_available():
    ext_module_cuda = cpp_extension.CUDAExtension(
        ".ops.ops_cuda",
        ["sparse_ops/ops/ops.cu"],
        extra_compile_args={'nvcc': []}
    )
    ext_modules.append(ext_module_cuda)

setup(
    name='sparse_ops',
    packages=['sparse_ops.ops'],
    ext_package='sparse_ops',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension,
              'install_lib': _CommandInstallCythonized
              }
)
