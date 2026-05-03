"""Tests for FrameAnnotator — visual output component."""
import pytest
import numpy as np

from ovp.core.types import (
    Detection, BoundingBox, Mask, SegmentedDetection, Track, FrameResult,
)
from ovp.viz.annotators import FrameAnnotator


# ============================================================
# Helpers
# ============================================================

def make_detection(label="person", x1=100, y1=100, x2=300, y2=400, score=0.85):
    return Detection(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        score=score,
        label=label,
    )


def make_mask(H=480, W=640, label="person"):
    """Make a mask with a rectangular True region."""
    data = np.zeros((H, W), dtype=np.bool_)
    data[100:300, 200:400] = True
    return Mask(data=data, score=0.9, label=label)


def make_track(detection, track_id=1):
    return Track(
        track_id=track_id,
        detection=detection,
        state="confirmed",
        age=0,
        frames_since_update=0,
        history=[],
    )


@pytest.fixture
def blank_image():
    """A 480x640 black image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def empty_frame_result():
    """FrameResult with no detections."""
    return FrameResult(
        frame_id=0,
        image_shape=(480, 640),
        detections=[],
        segmented=None,
        tracks=[],
        prompts=["person"],
        inference_times={},
    )


# ============================================================
# Construction
# ============================================================

class TestFrameAnnotatorConstruction:
    def test_default_construction(self):
        """Annotator should construct with defaults."""
        annotator = FrameAnnotator()
        assert annotator is not None
    
    def test_custom_opacity(self):
        """Custom mask_opacity should be accepted."""
        annotator = FrameAnnotator(mask_opacity=0.3)
        assert annotator is not None


# ============================================================
# Empty Cases
# ============================================================

class TestEmptyAnnotation:
    def test_empty_detections_returns_unchanged_image(self, blank_image, empty_frame_result):
        """No detections → output should be a copy of input (no drawing)."""
        annotator = FrameAnnotator()
        result = annotator.annotate(blank_image, empty_frame_result)
        
        # Pixel-perfect identity (içeriğin değişmemesi)
        assert np.array_equal(result, blank_image)
    
    def test_empty_returns_new_array(self, blank_image, empty_frame_result):
        """Even with empty detections, output should be a copy (not the same object)."""
        annotator = FrameAnnotator()
        result = annotator.annotate(blank_image, empty_frame_result)
        
        # Aynı içerik ama farklı array — caller mutate edebilsin
        assert result is not blank_image
    
    def test_output_shape_matches_input(self, blank_image, empty_frame_result):
        """Output shape should match input shape."""
        annotator = FrameAnnotator()
        result = annotator.annotate(blank_image, empty_frame_result)
        
        assert result.shape == blank_image.shape


# ============================================================
# With Detections (No Masks, No Tracks)
# ============================================================

class TestDetectionAnnotation:
    def test_detection_modifies_image(self, blank_image):
        """When there are detections, output should differ from input."""
        det = make_detection()
        fr = FrameResult(
            frame_id=0,
            image_shape=(480, 640),
            detections=[det],
            segmented=None,
            tracks=[],
            prompts=["person"],
            inference_times={},
        )
        
        annotator = FrameAnnotator()
        result = annotator.annotate(blank_image, fr)
        
        # Input ile output arasında pixel farkı olmalı (bbox/label çizildi)
        assert not np.array_equal(result, blank_image)
    
    def test_output_dtype_preserved(self, blank_image):
        """Output should remain uint8."""
        det = make_detection()
        fr = FrameResult(
            frame_id=0,
            image_shape=(480, 640),
            detections=[det],
            segmented=None,
            tracks=[],
            prompts=["person"],
            inference_times={},
        )
        
        annotator = FrameAnnotator()
        result = annotator.annotate(blank_image, fr)
        
        assert result.dtype == np.uint8
    
    def test_multiple_detections(self, blank_image):
        """Multiple detections should all be drawn."""
        dets = [
            make_detection(label="person", x1=50, y1=50, x2=200, y2=300),
            make_detection(label="car", x1=300, y1=100, x2=500, y2=400),
        ]
        fr = FrameResult(
            frame_id=0,
            image_shape=(480, 640),
            detections=dets,
            segmented=None,
            tracks=[],
            prompts=["person", "car"],
            inference_times={},
        )
        
        annotator = FrameAnnotator()
        result = annotator.annotate(blank_image, fr)
        
        # Pixel diff olmalı
        assert not np.array_equal(result, blank_image)


# ============================================================
# With Masks
# ============================================================

class TestMaskAnnotation:
    def test_mask_overlay_modifies_image(self, blank_image):
        """When segmented is provided, mask region should be drawn."""
        det = make_detection()
        mask = make_mask()
        sd = SegmentedDetection(detection=det, mask=mask)
        
        fr = FrameResult(
            frame_id=0,
            image_shape=(480, 640),
            detections=[det],
            segmented=[sd],
            tracks=[],
            prompts=["person"],
            inference_times={},
        )
        
        annotator = FrameAnnotator()
        result = annotator.annotate(blank_image, fr)
        
        # Mask alanında pixel değişmiş olmalı
        # Mask: y=100:300, x=200:400
        mask_region = result[100:300, 200:400]
        # En azından bazı pixeller siyah (0,0,0) olmamalı (mask çizildi)
        assert mask_region.sum() > 0


# ============================================================
# With Tracks
# ============================================================

class TestTrackAwareAnnotation:
    def test_with_tracks_modifies_image(self, blank_image):
        """When tracks exist, output should differ (track ID prefix in label)."""
        det = make_detection()
        track = make_track(det, track_id=42)
        
        fr = FrameResult(
            frame_id=0,
            image_shape=(480, 640),
            detections=[det],
            segmented=None,
            tracks=[track],
            prompts=["person"],
            inference_times={},
        )
        
        annotator = FrameAnnotator()
        result = annotator.annotate(blank_image, fr)
        
        # Çizim yapıldı
        assert not np.array_equal(result, blank_image)
    
    def test_without_tracks_still_works(self, blank_image):
        """No tracks → fallback to standard label format, no crash."""
        det = make_detection()
        fr = FrameResult(
            frame_id=0,
            image_shape=(480, 640),
            detections=[det],
            segmented=None,
            tracks=[],  # boş tracks
            prompts=["person"],
            inference_times={},
        )
        
        annotator = FrameAnnotator()
        # Crash etmemeli
        result = annotator.annotate(blank_image, fr)
        assert result.shape == blank_image.shape


# ============================================================
# Full Pipeline (Detections + Masks + Tracks)
# ============================================================

class TestFullAnnotation:
    def test_all_components_together(self, blank_image):
        """Detections + masks + tracks together should not crash."""
        det = make_detection()
        mask = make_mask()
        sd = SegmentedDetection(detection=det, mask=mask)
        track = make_track(det, track_id=1)
        
        fr = FrameResult(
            frame_id=0,
            image_shape=(480, 640),
            detections=[det],
            segmented=[sd],
            tracks=[track],
            prompts=["person"],
            inference_times={},
        )
        
        annotator = FrameAnnotator()
        result = annotator.annotate(blank_image, fr)
        
        # Hepsi birleşince image modifiye olmalı
        assert not np.array_equal(result, blank_image)
        assert result.shape == blank_image.shape
        assert result.dtype == np.uint8