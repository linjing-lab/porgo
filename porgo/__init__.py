# Copyright (c) 2024 linjing-lab

import sys

from .runtime import glos
from ._version import __version__

if sys.version_info < (3, 7, 0):
    raise OSError(f'porgo requires Python >=3.7, but yours is {sys.version}')