from typing import Dict, Type, Any

ELEMENT_REGISTRY: Dict[str, Type[Any]] = {}
KERNEL_REGISTRY: Dict[str, Type[Any]] = {}
INJECTOR_REGISTRY: Dict[str, Type[Any]] = {}

def register_element(name: str):
    """A decorator to register an Element with a given string identifier."""
    def decorator(cls: Type[Any]):
        if name in ELEMENT_REGISTRY:
            raise ValueError(f"Element '{name}' is already registered.")
        ELEMENT_REGISTRY[name] = cls
        return cls
    return decorator

def get_element(name: str) -> Type[Any]:
    """Looks up an Element in the registry by its identifier."""
    if name not in ELEMENT_REGISTRY:
        raise KeyError(f"No Element registered for '{name}'. Available: {list(ELEMENT_REGISTRY.keys())}")
    return ELEMENT_REGISTRY[name]

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