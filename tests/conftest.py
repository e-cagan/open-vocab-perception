"""Shared pytest fixtures for the test suite."""

import numpy as np
import pytest

from ovp.core.types import (
    BoundingBox,
    Detection,
    FrameResult,
    Mask,
    SegmentedDetection,
)


@pytest.fixture
def sample_bbox() -> BoundingBox:
    """A simple valid bounding box."""
    return BoundingBox(x1=10, y1=20, x2=100, y2=200)


@pytest.fixture
def sample_detection(sample_bbox) -> Detection:
    """A simple valid detection (person, score 0.85)."""
    return Detection(bbox=sample_bbox, score=0.85, label="person")


@pytest.fixture
def sample_mask() -> Mask:
    """A simple valid mask (480x640 with a rectangular region True)."""
    data = np.zeros((480, 640), dtype=np.bool_)
    data[100:300, 200:400] = True
    return Mask(data=data, score=0.9, label="person")


@pytest.fixture
def sample_segmented(sample_detection, sample_mask) -> SegmentedDetection:
    """A SegmentedDetection combining sample_detection and sample_mask."""
    return SegmentedDetection(detection=sample_detection, mask=sample_mask)


@pytest.fixture
def sample_image_np() -> np.ndarray:
    """A blank RGB image (480x640x3, uint8, all zeros)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def empty_frame_result() -> FrameResult:
    """A FrameResult with no detections (empty scene)."""
    return FrameResult(
        frame_id=0,
        image_shape=(480, 640),
        detections=[],
        segmented=None,
        tracks=[],
        prompts=["person"],
        inference_times={},
    )
