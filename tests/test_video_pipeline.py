"""Tests for VideoPipeline orchestration logic, using mock components."""

import numpy as np
import pytest

from ovp.core.interfaces import BaseDetector, BaseSegmenter, BaseTracker
from ovp.core.types import (
    BoundingBox,
    Detection,
    FrameResult,
    Mask,
    Track,
)
from ovp.pipeline.video_pipeline import VideoPipeline

# ============================================================
# Mocks
# ============================================================


class MockDetector(BaseDetector):
    def __init__(self, return_detections=None, device="cpu"):
        self._return_detections = return_detections or []
        self._device = device
        self.detect_call_count = 0

    @property
    def device(self):
        return self._device

    @property
    def name(self):
        return "mock_detector"

    def detect(self, image, prompts, threshold=None):
        self.detect_call_count += 1
        return list(self._return_detections)  # copy to be safe


class MockSegmenter(BaseSegmenter):
    def __init__(self, device="cpu"):
        self._device = device
        self.segment_call_count = 0

    @property
    def device(self):
        return self._device

    @property
    def name(self):
        return "mock_segmenter"

    def segment(self, image, boxes):
        self.segment_call_count += 1
        H, W = image.shape[:2]
        return [Mask(data=np.zeros((H, W), dtype=np.bool_), score=0.95, label=None) for _ in boxes]


class MockTracker(BaseTracker):
    def __init__(self):
        self._next_id = 1
        self.update_call_count = 0
        self.reset_call_count = 0

    @property
    def name(self):
        return "mock_tracker"

    def update(self, detections):
        self.update_call_count += 1
        tracks = []
        for det in detections:
            tracks.append(
                Track(
                    track_id=self._next_id,
                    detection=det,
                    state="confirmed",
                    age=0,
                    frames_since_update=0,
                    history=[],
                )
            )
            self._next_id += 1
        return tracks

    def reset(self):
        self.reset_call_count += 1
        self._next_id = 1


# ============================================================
# Helpers
# ============================================================


def make_detection(label="person"):
    return Detection(
        bbox=BoundingBox(x1=10, y1=20, x2=100, y2=200),
        score=0.85,
        label=label,
    )


def frame_stream(n_frames: int, shape=(480, 640, 3)):
    """Generate n blank frames for testing."""
    for _ in range(n_frames):
        yield np.zeros(shape, dtype=np.uint8)


@pytest.fixture
def blank_frames_30():
    """30 blank frames as a list (rewindable for assertions)."""
    return [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(30)]


# ============================================================
# Construction
# ============================================================


class TestVideoPipelineConstruction:
    def test_with_detector_only(self):
        pipeline = VideoPipeline(detector=MockDetector())
        assert pipeline is not None

    def test_with_all_components(self):
        pipeline = VideoPipeline(
            detector=MockDetector(),
            segmenter=MockSegmenter(),
            tracker=MockTracker(),
        )
        assert pipeline is not None

    def test_default_keyframe_interval(self):
        pipeline = VideoPipeline(detector=MockDetector())
        # keyframe_interval default'u accessible olmalı
        assert pipeline._keyframe_interval > 0


# ============================================================
# Output Format
# ============================================================


class TestVideoPipelineOutput:
    def test_yields_tuples(self, blank_frames_30):
        """run_video should yield (frame, FrameResult) tuples."""
        pipeline = VideoPipeline(detector=MockDetector())

        for item in pipeline.run_video(iter(blank_frames_30[:5]), prompts=["person"]):
            assert isinstance(item, tuple)
            assert len(item) == 2
            frame, result = item
            assert isinstance(frame, np.ndarray)
            assert isinstance(result, FrameResult)

    def test_yields_one_per_frame(self, blank_frames_30):
        """One yield per input frame."""
        pipeline = VideoPipeline(detector=MockDetector())

        results = list(pipeline.run_video(iter(blank_frames_30[:10]), prompts=["person"]))

        assert len(results) == 10

    def test_frame_ids_sequential(self, blank_frames_30):
        """frame_id should increment from 0."""
        pipeline = VideoPipeline(detector=MockDetector())

        results = list(pipeline.run_video(iter(blank_frames_30[:5]), prompts=["person"]))
        frame_ids = [r.frame_id for _, r in results]

        assert frame_ids == [0, 1, 2, 3, 4]


# ============================================================
# Keyframe Strategy
# ============================================================


class TestVideoPipelineKeyframes:
    def test_detector_called_only_on_keyframes(self, blank_frames_30):
        """With keyframe_interval=10, detector runs on frame 0, 10, 20."""
        detector = MockDetector(return_detections=[make_detection()])
        pipeline = VideoPipeline(detector=detector, keyframe_interval=10)

        list(pipeline.run_video(iter(blank_frames_30), prompts=["person"]))

        # 30 frame, k=10 → frames 0, 10, 20 = 3 keyframes
        assert detector.detect_call_count == 3

    def test_segmenter_called_only_on_keyframes(self, blank_frames_30):
        """Segmenter same pattern."""
        detector = MockDetector(return_detections=[make_detection()])
        segmenter = MockSegmenter()
        pipeline = VideoPipeline(
            detector=detector,
            segmenter=segmenter,
            keyframe_interval=10,
        )

        list(pipeline.run_video(iter(blank_frames_30), prompts=["person"]))

        # 3 keyframe → 3 segmenter call (her keyframe'de detection vardı)
        assert segmenter.segment_call_count == 3

    def test_keyframe_interval_one_runs_every_frame(self, blank_frames_30):
        """keyframe_interval=1 means every frame is a keyframe."""
        detector = MockDetector(return_detections=[make_detection()])
        pipeline = VideoPipeline(detector=detector, keyframe_interval=1)

        list(pipeline.run_video(iter(blank_frames_30[:5]), prompts=["person"]))

        assert detector.detect_call_count == 5

    def test_regular_frames_reuse_keyframe_result(self, blank_frames_30):
        """Regular frames should reuse last keyframe's detections."""
        detector = MockDetector(return_detections=[make_detection(label="cat")])
        pipeline = VideoPipeline(detector=detector, keyframe_interval=5)

        results = list(pipeline.run_video(iter(blank_frames_30[:10]), prompts=["cat"]))

        # Hem keyframe hem regular frame'lerde detection olmalı (cache)
        for _, result in results:
            assert len(result.detections) == 1
            assert result.detections[0].label == "cat"


# ============================================================
# Tracker Lifecycle
# ============================================================


class TestVideoPipelineTracker:
    def test_tracker_reset_called_at_start(self, blank_frames_30):
        """Tracker should be reset when a new video starts."""
        tracker = MockTracker()
        pipeline = VideoPipeline(detector=MockDetector(), tracker=tracker)

        list(pipeline.run_video(iter(blank_frames_30[:3]), prompts=["person"]))

        # Yeni video başlarken reset edilmeli
        assert tracker.reset_call_count >= 1

    def test_tracker_update_called_on_keyframes(self, blank_frames_30):
        """Tracker.update should be called on keyframes (with detections)."""
        detector = MockDetector(return_detections=[make_detection()])
        tracker = MockTracker()
        pipeline = VideoPipeline(
            detector=detector,
            tracker=tracker,
            keyframe_interval=10,
        )

        list(pipeline.run_video(iter(blank_frames_30), prompts=["person"]))

        # 3 keyframe = 3 update call (en azından)
        assert tracker.update_call_count >= 3

    def test_no_tracker_yields_empty_tracks(self, blank_frames_30):
        """Without tracker, result.tracks should be empty list."""
        detector = MockDetector(return_detections=[make_detection()])
        pipeline = VideoPipeline(detector=detector)  # no tracker

        results = list(pipeline.run_video(iter(blank_frames_30[:5]), prompts=["person"]))

        for _, result in results:
            assert result.tracks == []


# ============================================================
# Validation
# ============================================================


class TestVideoPipelineValidation:
    def test_empty_prompts_raises(self, blank_frames_30):
        """Empty prompts should raise ValueError."""
        pipeline = VideoPipeline(detector=MockDetector())

        with pytest.raises(ValueError):
            # Generator must be consumed for validation to trigger
            list(pipeline.run_video(iter(blank_frames_30[:1]), prompts=[]))

    def test_empty_video_yields_nothing(self):
        """Empty frame stream should yield no results."""
        pipeline = VideoPipeline(detector=MockDetector())

        results = list(pipeline.run_video(iter([]), prompts=["person"]))

        assert results == []
