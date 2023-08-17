from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
from torch.utils.cpp_extension import CUDA_HOME
import os
import re
from torch import cuda


'''
Hack to remove lib.*.so from the output .so files.
'''
from distutils.command.install_lib import install_lib as _install_lib


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


ext_modules = []

ext_module_cpp = CppExtension(
    'ops.lib.ops_cc',
    ['ops/lib/ops.cc'],
    extra_compile_args=['-fopenmp', '-Wall', '-Werror']
)

ext_modules = [ext_module_cpp]

if cuda.is_available():
    ext_module_cuda = CUDAExtension(
        "ops.lib.ops_cuda",
        ["ops/lib/ops.cu"],
        extra_compile_args={'nvcc': []}
    )
    ext_modules.append(ext_module_cuda)

setup(
    name='ops',
    packages=['ops.lib'],
    platforms='Any',
    classifiers=[],
    ext_package='',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension,
              'install_lib': _CommandInstallCythonized
              })
