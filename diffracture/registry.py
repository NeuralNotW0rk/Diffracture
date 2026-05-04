from typing import Dict, Type, Any

PRISM_REGISTRY: Dict[str, Type[Any]] = {}
KERNEL_REGISTRY: Dict[str, Type[Any]] = {}
INJECTOR_REGISTRY: Dict[str, Type[Any]] = {}

def register_prism(name: str):
    """A decorator to register a Prism with a given string identifier."""
    def decorator(cls: Type[Any]):
        if name in PRISM_REGISTRY:
            raise ValueError(f"Prism '{name}' is already registered.")
        PRISM_REGISTRY[name] = cls
        return cls
    return decorator

def get_prism(name: str) -> Type[Any]:
    """Looks up a Prism in the registry by its identifier."""
    if name not in PRISM_REGISTRY:
        raise KeyError(f"No Prism registered for '{name}'. Available: {list(PRISM_REGISTRY.keys())}")
    return PRISM_REGISTRY[name]

def register_kernel(name: str):
    """A decorator to register a Kernel with a given string identifier."""
    def decorator(cls: Type[Any]):
        if name in KERNEL_REGISTRY:
            raise ValueError(f"Kernel '{name}' is already registered.")
        KERNEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_kernel(name: str) -> Type[Any]:
    if name not in KERNEL_REGISTRY:
        raise KeyError(f"No Kernel registered for '{name}'. Available: {list(KERNEL_REGISTRY.keys())}")
    return KERNEL_REGISTRY[name]

def register_injector(name: str):
    """A decorator to register an Injector with a given string identifier."""
    def decorator(cls: Type[Any]):
        if name in INJECTOR_REGISTRY:
            raise ValueError(f"Injector '{name}' is already registered.")
        INJECTOR_REGISTRY[name] = cls
        return cls
    return decorator

def get_injector(name: str) -> Type[Any]:
    if name not in INJECTOR_REGISTRY:
        raise KeyError(f"No Injector registered for '{name}'. Available: {list(INJECTOR_REGISTRY.keys())}")
    return INJECTOR_REGISTRY[name]