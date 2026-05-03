"""Tests for ImagePipeline orchestration logic, using mock components."""
import pytest
import numpy as np

from ovp.core.types import (
    Detection, BoundingBox, Mask, SegmentedDetection, FrameResult,
)
from ovp.core.interfaces import BaseDetector, BaseSegmenter
from ovp.pipeline.image_pipeline import ImagePipeline


# ============================================================
# Mocks
# ============================================================

class MockDetector(BaseDetector):
    """A fake detector that returns pre-configured detections."""
    
    def __init__(self, return_detections=None, device="cpu"):
        self._return_detections = return_detections or []
        self._device = device
        self.detect_call_count = 0
        self.last_prompts = None
        self.last_threshold = None
    
    @property
    def device(self):
        return self._device
    
    @property
    def name(self):
        return "mock_detector"
    
    def detect(self, image, prompts, threshold=None):
        self.detect_call_count += 1
        self.last_prompts = prompts
        self.last_threshold = threshold
        return self._return_detections


class MockSegmenter(BaseSegmenter):
    """A fake segmenter producing one mask per input bbox."""
    
    def __init__(self, device="cpu"):
        self._device = device
        self.segment_call_count = 0
        self.last_boxes = None
    
    @property
    def device(self):
        return self._device
    
    @property
    def name(self):
        return "mock_segmenter"
    
    def segment(self, image, boxes):
        self.segment_call_count += 1
        self.last_boxes = boxes
        H, W = image.shape[:2]
        return [
            Mask(
                data=np.zeros((H, W), dtype=np.bool_),
                score=0.95,
                label=None,
            )
            for _ in boxes
        ]


# ============================================================
# Helpers
# ============================================================

def make_detection(x1=10, y1=20, x2=100, y2=200, score=0.85, label="person"):
    return Detection(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        score=score,
        label=label,
    )


@pytest.fixture
def blank_image():
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ============================================================
# Construction
# ============================================================

class TestImagePipelineConstruction:
    def test_with_detector_only(self):
        """Pipeline can be constructed without segmenter."""
        detector = MockDetector()
        pipeline = ImagePipeline(detector=detector)
        assert pipeline is not None
    
    def test_with_detector_and_segmenter(self):
        """Pipeline accepts both detector and segmenter."""
        pipeline = ImagePipeline(
            detector=MockDetector(),
            segmenter=MockSegmenter(),
        )
        assert pipeline is not None


# ============================================================
# Run — Happy Paths
# ============================================================

class TestImagePipelineRun:
    def test_returns_frame_result(self, blank_image):
        """run() should always return a FrameResult."""
        pipeline = ImagePipeline(detector=MockDetector())
        result = pipeline.run(blank_image, prompts=["person"])
        
        assert isinstance(result, FrameResult)
    
    def test_detector_called_once(self, blank_image):
        """run() should call detect exactly once."""
        detector = MockDetector(return_detections=[make_detection()])
        pipeline = ImagePipeline(detector=detector)
        
        pipeline.run(blank_image, prompts=["person"])
        
        assert detector.detect_call_count == 1
    
    def test_prompts_passed_to_detector(self, blank_image):
        """Prompts should reach the detector unchanged."""
        detector = MockDetector()
        pipeline = ImagePipeline(detector=detector)
        
        pipeline.run(blank_image, prompts=["person", "car"])
        
        assert detector.last_prompts == ["person", "car"]
    
    def test_threshold_passed_to_detector(self, blank_image):
        """detector_threshold should propagate to detector.detect()."""
        detector = MockDetector()
        pipeline = ImagePipeline(detector=detector)
        
        pipeline.run(blank_image, prompts=["person"], detector_threshold=0.5)
        
        assert detector.last_threshold == 0.5
    
    def test_detections_in_result(self, blank_image):
        """Detector output should appear in result.detections."""
        det_objects = [make_detection(label="cat"), make_detection(label="dog")]
        detector = MockDetector(return_detections=det_objects)
        pipeline = ImagePipeline(detector=detector)
        
        result = pipeline.run(blank_image, prompts=["cat", "dog"])
        
        assert len(result.detections) == 2
        labels = [d.label for d in result.detections]
        assert "cat" in labels
        assert "dog" in labels
    
    def test_image_shape_in_result(self, blank_image):
        """result.image_shape should match input image."""
        pipeline = ImagePipeline(detector=MockDetector())
        result = pipeline.run(blank_image, prompts=["person"])
        
        assert result.image_shape == (480, 640)
    
    def test_inference_times_recorded(self, blank_image):
        """result.inference_times should contain timing info."""
        pipeline = ImagePipeline(detector=MockDetector())
        result = pipeline.run(blank_image, prompts=["person"])
        
        assert "detector" in result.inference_times
        assert result.inference_times["detector"] > 0


# ============================================================
# Segmenter Integration
# ============================================================

class TestImagePipelineSegmenter:
    def test_segmenter_called_when_detections_exist(self, blank_image):
        """If detector finds objects, segmenter should be called."""
        detector = MockDetector(return_detections=[make_detection()])
        segmenter = MockSegmenter()
        pipeline = ImagePipeline(detector=detector, segmenter=segmenter)
        
        pipeline.run(blank_image, prompts=["person"])
        
        assert segmenter.segment_call_count == 1
    
    def test_segmenter_receives_detection_boxes(self, blank_image):
        """Segmenter should receive bboxes from detector output."""
        det = make_detection(x1=50, y1=60, x2=200, y2=300)
        detector = MockDetector(return_detections=[det])
        segmenter = MockSegmenter()
        pipeline = ImagePipeline(detector=detector, segmenter=segmenter)
        
        pipeline.run(blank_image, prompts=["person"])
        
        assert len(segmenter.last_boxes) == 1
        assert segmenter.last_boxes[0].x1 == 50
        assert segmenter.last_boxes[0].x2 == 200
    
    def test_segmenter_skipped_when_no_detections(self, blank_image):
        """If detector returns empty, segmenter should NOT be called."""
        detector = MockDetector(return_detections=[])
        segmenter = MockSegmenter()
        pipeline = ImagePipeline(detector=detector, segmenter=segmenter)
        
        pipeline.run(blank_image, prompts=["person"])
        
        # Empty detection → segmenter doesn't called (waste of time)
        assert segmenter.segment_call_count == 0
    
    def test_segmented_in_result_when_segmenter_used(self, blank_image):
        """result.segmented should contain SegmentedDetections."""
        detector = MockDetector(return_detections=[make_detection()])
        segmenter = MockSegmenter()
        pipeline = ImagePipeline(detector=detector, segmenter=segmenter)
        
        result = pipeline.run(blank_image, prompts=["person"])
        
        assert result.segmented is not None
        assert len(result.segmented) == 1
        assert isinstance(result.segmented[0], SegmentedDetection)
    
    def test_segmented_none_without_segmenter(self, blank_image):
        """If pipeline has no segmenter, result.segmented should be None."""
        detector = MockDetector(return_detections=[make_detection()])
        pipeline = ImagePipeline(detector=detector)  # no segmenter
        
        result = pipeline.run(blank_image, prompts=["person"])
        
        assert result.segmented is None
    
    def test_segmenter_inference_time_recorded(self, blank_image):
        """When segmenter runs, its time should be in inference_times."""
        detector = MockDetector(return_detections=[make_detection()])
        segmenter = MockSegmenter()
        pipeline = ImagePipeline(detector=detector, segmenter=segmenter)
        
        result = pipeline.run(blank_image, prompts=["person"])
        
        assert "segmenter" in result.inference_times
        assert result.inference_times["segmenter"] > 0


# ============================================================
# Validation & Edge Cases
# ============================================================

class TestImagePipelineValidation:
    def test_empty_prompts_raises(self, blank_image):
        """Empty prompts should raise ValueError."""
        pipeline = ImagePipeline(detector=MockDetector())
        
        with pytest.raises(ValueError):
            pipeline.run(blank_image, prompts=[])
    
    def test_zero_detections_returns_valid_frame_result(self, blank_image):
        """Pipeline must handle empty detections gracefully."""
        detector = MockDetector(return_detections=[])
        pipeline = ImagePipeline(detector=detector)
        
        result = pipeline.run(blank_image, prompts=["unicorn"])
        
        assert isinstance(result, FrameResult)
        assert len(result.detections) == 0
        assert result.segmented is None  # no detections → no segmented