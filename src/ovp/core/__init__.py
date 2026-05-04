"""
Core abstractions for the OVP pipeline:
- types: Pydantic data models (Detection, Mask, Track, FrameResult, ...)
- interfaces: Abstract base classes (BaseDetector, BaseSegmenter, BaseTracker)
- registry: Type-safe component registries
"""

from ovp.core.interfaces import (
    BaseDetector,
    BaseSegmenter,
    BaseTracker,
)
from ovp.core.registry import (
    DETECTOR_REGISTRY,
    SEGMENTER_REGISTRY,
    TRACKER_REGISTRY,
    Registry,
)
from ovp.core.types import (
    BoundingBox,
    Detection,
    FrameResult,
    Mask,
    SegmentedDetection,
    Track,
)

__all__ = [
    # types
    "BoundingBox",
    "Detection",
    "Mask",
    "SegmentedDetection",
    "Track",
    "FrameResult",
    # interfaces
    "BaseDetector",
    "BaseSegmenter",
    "BaseTracker",
    # registry
    "Registry",
    "DETECTOR_REGISTRY",
    "SEGMENTER_REGISTRY",
    "TRACKER_REGISTRY",
]
