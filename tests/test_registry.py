"""Tests for the type-safe component registry."""

import pytest

from ovp.core.registry import Registry


# Mock base class for isolated testing — gerçek Detector/Segmenter import etmiyoruz
class FakeBase:
    """A minimal class for testing the generic Registry."""

    def __init__(self, name: str = "default"):
        self.name = name


class TestRegistry:
    def test_empty_registry(self):
        """New registry should be empty."""
        reg = Registry[FakeBase]("fake")
        assert len(reg) == 0
        assert reg.keys() == []

    def test_register_decorator_pattern(self):
        """Decorator usage: @reg.register('key')."""
        reg = Registry[FakeBase]("fake")

        @reg.register("foo")
        class Foo(FakeBase):
            pass

        assert "foo" in reg
        assert reg.get("foo") is Foo

    def test_register_direct_call_pattern(self):
        """Direct usage: reg.register('key', cls)."""
        reg = Registry[FakeBase]("fake")
        reg.register("foo", FakeBase)

        assert "foo" in reg
        assert reg.get("foo") is FakeBase

    def test_register_decorator_returns_class(self):
        """Decorator should return the class unchanged (preserves identity)."""
        reg = Registry[FakeBase]("fake")

        decorated = reg.register("foo")(FakeBase)

        assert decorated is FakeBase

    def test_contains(self):
        """__contains__ should work for registered keys."""
        reg = Registry[FakeBase]("fake")
        reg.register("foo", FakeBase)

        assert "foo" in reg
        assert "bar" not in reg

    def test_keys_sorted(self):
        """keys() should return sorted list."""
        reg = Registry[FakeBase]("fake")
        reg.register("zebra", FakeBase)
        reg.register("apple", FakeBase)
        reg.register("mango", FakeBase)

        # Senin implementation sorted dönüyor
        assert reg.keys() == ["apple", "mango", "zebra"]

    def test_len(self):
        """__len__ should reflect number of registered classes."""
        reg = Registry[FakeBase]("fake")
        assert len(reg) == 0

        reg.register("foo", FakeBase)
        assert len(reg) == 1

        reg.register("bar", FakeBase)
        assert len(reg) == 2

    def test_duplicate_registration_raises(self):
        """Registering same key twice should raise ValueError."""
        reg = Registry[FakeBase]("fake")
        reg.register("foo", FakeBase)

        with pytest.raises(ValueError):
            reg.register("foo", FakeBase)

    def test_duplicate_error_mentions_existing(self):
        """ValueError message should mention what was already registered."""
        reg = Registry[FakeBase]("fake")
        reg.register("foo", FakeBase)

        with pytest.raises(ValueError) as exc_info:
            reg.register("foo", FakeBase)

        # Mesaj zaten kayıtlı class'ı belirtmeli
        msg = str(exc_info.value)
        assert "foo" in msg
        assert "FakeBase" in msg

    def test_get_missing_key_raises(self):
        """get() with unregistered key should raise KeyError."""
        reg = Registry[FakeBase]("fake")

        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_create_missing_key_raises(self):
        """create() with unregistered key should raise KeyError."""
        reg = Registry[FakeBase]("fake")

        with pytest.raises(KeyError):
            reg.create("nonexistent")

    def test_create_error_lists_available_keys(self):
        """create() error message should list available keys for discoverability."""
        reg = Registry[FakeBase]("fake")
        reg.register("foo", FakeBase)
        reg.register("bar", FakeBase)

        with pytest.raises(KeyError) as exc_info:
            reg.create("baz")

        msg = str(exc_info.value)
        # Available keys gösterilmeli
        assert "foo" in msg
        assert "bar" in msg

    def test_create_with_no_registered_lists_none(self):
        """create() error on empty registry should say '(none)'."""
        reg = Registry[FakeBase]("fake")

        with pytest.raises(KeyError) as exc_info:
            reg.create("anything")

        msg = str(exc_info.value)
        assert "(none)" in msg

    def test_create_instantiates_with_kwargs(self):
        """create() should instantiate with provided kwargs."""
        reg = Registry[FakeBase]("fake")
        reg.register("foo", FakeBase)

        instance = reg.create("foo", name="custom")

        assert isinstance(instance, FakeBase)
        assert instance.name == "custom"

    def test_create_instantiates_with_defaults(self):
        """create() without kwargs should use class defaults."""
        reg = Registry[FakeBase]("fake")
        reg.register("foo", FakeBase)

        instance = reg.create("foo")

        assert instance.name == "default"

    def test_repr(self):
        """__repr__ should be informative."""
        reg = Registry[FakeBase]("fake")
        reg.register("foo", FakeBase)

        rep = repr(reg)
        assert "fake" in rep
        assert "foo" in rep


class TestModuleRegistries:
    """Test the module-level singleton registries (real components)."""

    def test_detector_registry_has_grounding_dino(self):
        # Side-effect import for decorator registration
        import ovp.detectors.grounding_dino  # noqa: F401
        from ovp.core.registry import DETECTOR_REGISTRY

        assert "grounding_dino" in DETECTOR_REGISTRY

    def test_segmenter_registry_has_sam2(self):
        import ovp.segmenters.sam2  # noqa: F401
        from ovp.core.registry import SEGMENTER_REGISTRY

        assert "sam2" in SEGMENTER_REGISTRY

    def test_tracker_registry_has_bytetrack(self):
        import ovp.trackers.bytetrack  # noqa: F401
        from ovp.core.registry import TRACKER_REGISTRY

        assert "bytetrack" in TRACKER_REGISTRY

    def test_registries_are_distinct_singletons(self):
        """The three module-level registries are separate instances."""
        from ovp.core.registry import (
            DETECTOR_REGISTRY,
            SEGMENTER_REGISTRY,
            TRACKER_REGISTRY,
        )

        assert DETECTOR_REGISTRY is not SEGMENTER_REGISTRY
        assert SEGMENTER_REGISTRY is not TRACKER_REGISTRY
        assert DETECTOR_REGISTRY is not TRACKER_REGISTRY
