import gc
import pdb
import sys

import glog
import torch


def clean():
    gc.collect()
    torch.cuda.empty_cache()
