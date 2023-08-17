import os
import sysconfig
import torch

_HERE = os.path.realpath(os.path.dirname(__file__))
EXT_SUFFIX = sysconfig.get_config_var('EXT_SUFFIX')

if (torch.cuda.is_available()):
    torch.ops.load_library(_HERE + '/ops_cuda.so')

torch.ops.load_library(_HERE + '/ops_cc.so')