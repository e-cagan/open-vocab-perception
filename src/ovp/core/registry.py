"""
Registry for component implementations.

Provides a centralized way to register and instantiate concrete
implementations of BaseDetector, BaseSegmenter, and BaseTracker
by string identifiers, decoupling configuration from code.
"""

from typing import Generic, TypeVar, Callable
from ovp.core.interfaces import BaseDetector, BaseSegmenter, BaseTracker

# Type can be anything
T = TypeVar("T")


class Registry(Generic[T]):
    """
    A type-safe registry mapping string keys to component classes.
    
    Subclasses of a common base class can register themselves under
    string identifiers, then be instantiated by name with arbitrary
    constructor kwargs.
    """
    
    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: dict[str, type[T]] = {}
    
    def register(
        self, 
        key: str,
        cls: type[T] | None = None,
    ) -> Callable[[type[T]], type[T]] | type[T]:
        """
        Register a class under the given key.
        
        Can be used as a decorator:
            @REGISTRY.register("foo")
            class FooImpl(Base): ...
        
        Or as a direct call:
            REGISTRY.register("foo", FooImpl)
        """
        def _do_register(cls_inner: type[T]) -> type[T]:
            if key in self._registry:
                raise ValueError(
                    f"{self._name}: key '{key}' is already registered "
                    f"to {self._registry[key].__name__}"
                )
            self._registry[key] = cls_inner
            return cls_inner
        
        if cls is None:
            # Decorator usage
            return _do_register
        # Direct call usage
        return _do_register(cls)
    
    def create(self, key: str, **kwargs) -> T:
        """Instantiate the registered class by key with given kwargs."""
        if key not in self._registry:
            available = ", ".join(sorted(self._registry.keys())) or "(none)"
            raise KeyError(
                f"{self._name}: '{key}' not registered. Available: {available}"
            )
        return self._registry[key](**kwargs)
    
    def get(self, key: str) -> type[T]:
        """Return the registered class (without instantiating)."""
        if key not in self._registry:
            raise KeyError(f"{self._name}: '{key}' not registered")
        return self._registry[key]
    
    def keys(self) -> list[str]:
        """List all registered keys."""
        return sorted(self._registry.keys())
    
    def __contains__(self, key: str) -> bool:
        return key in self._registry
    
    def __len__(self) -> int:
        return len(self._registry)
    
    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, keys={self.keys()})"


# Global registries — one per component family
DETECTOR_REGISTRY: Registry[BaseDetector] = Registry("DETECTOR_REGISTRY")
SEGMENTER_REGISTRY: Registry[BaseSegmenter] = Registry("SEGMENTER_REGISTRY")
TRACKER_REGISTRY: Registry[BaseTracker] = Registry("TRACKER_REGISTRY")