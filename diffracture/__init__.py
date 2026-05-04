"""
Diffracture: A modular library for model manipulation and injection.
"""

import importlib
import pkgutil
from pathlib import Path

# Expose registry functions at the package level
from .registry import get_prism, get_kernel, get_injector

# Main high-level API classes
from .topology.lattice import Lattice
from .actant import Actant

# Auto-discover and import all modules in the package to trigger @register decorators
_package_dir = str(Path(__file__).resolve().parent)
for _, _module_name, _ in pkgutil.walk_packages([_package_dir], prefix=__name__ + "."):
    importlib.import_module(_module_name)