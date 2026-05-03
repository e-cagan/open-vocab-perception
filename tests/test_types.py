"""Tests for core type system (Pydantic models)."""

import pytest
import numpy as np
from pydantic import ValidationError

from ovp.core.types import (
    BoundingBox, Detection, Mask, SegmentedDetection, Track, FrameResult
)


class TestBoundingBox:
    def test_valid_bbox(self):
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        assert bbox.x1 == 10
        assert bbox.format == "xyxy"
    
    def test_invalid_corner_order_raises(self):
        with pytest.raises(ValidationError):
            BoundingBox(x1=100, y1=20, x2=10, y2=200)
        with pytest.raises(ValidationError):
            BoundingBox(x1=10, y1=200, x2=100, y2=20)
    
    def test_iou_identical_boxes(self):
        a = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        b = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        assert a.iou(b) == 1.0
    
    def test_iou_no_overlap(self):
        a = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        b = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        assert a.iou(b) == 0.0
    
    def test_iou_half_overlap(self):
        a = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        b = BoundingBox(x1=50, y1=0, x2=150, y2=100)
        # intersection: 50x100 = 5000
        # union: 10000 + 10000 - 5000 = 15000
        # iou: 1/3
        assert abs(a.iou(b) - 1/3) < 0.001
    
    def test_iou_contained(self):
        # B fully inside A
        a = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        b = BoundingBox(x1=25, y1=25, x2=75, y2=75)
        # intersection: 50x50 = 2500
        # union: 10000 + 2500 - 2500 = 10000
        # iou: 0.25
        assert abs(a.iou(b) - 0.25) < 0.001


class TestDetection:
    def test_valid_detection(self, sample_bbox):
        d = Detection(bbox=sample_bbox, score=0.85, label="person")
        assert d.score == 0.85
        assert d.embedding is None
        assert d.metadata == {}
    
    def test_score_above_one_raises(self, sample_bbox):
        with pytest.raises(ValidationError):
            Detection(bbox=sample_bbox, score=1.5, label="person")
    
    def test_score_below_zero_raises(self, sample_bbox):
        with pytest.raises(ValidationError):
            Detection(bbox=sample_bbox, score=-0.1, label="person")
    
    def test_score_at_boundaries_valid(self, sample_bbox):
        # 0.0 ve 1.0 sınır değerleri valid olmalı
        d_zero = Detection(bbox=sample_bbox, score=0.0, label="person")
        d_one = Detection(bbox=sample_bbox, score=1.0, label="person")
        assert d_zero.score == 0.0
        assert d_one.score == 1.0


class TestMask:
    def test_valid_mask(self):
        data = np.zeros((100, 200), dtype=np.bool_)
        m = Mask(data=data, score=0.9, label="cat")
        assert m.data.shape == (100, 200)
        assert m.data.dtype == np.bool_
    
    def test_wrong_dtype_rejected(self):
        data = np.zeros((100, 200), dtype=np.uint8)
        with pytest.raises(ValidationError):
            Mask(data=data, score=0.9, label="cat")
    
    def test_wrong_shape_rejected(self):
        # 3D array should fail (mask is 2D)
        data = np.zeros((100, 200, 3), dtype=np.bool_)
        with pytest.raises(ValidationError):
            Mask(data=data, score=0.9, label="cat")


class TestTrack:
    def test_valid_track(self, sample_detection):
        t = Track(
            track_id=1,
            detection=sample_detection,
            state="confirmed",
            age=5,
            frames_since_update=2,
        )
        assert t.track_id == 1
        assert t.state == "confirmed"
    
    def test_frames_since_update_exceeds_age_raises(self, sample_detection):
        with pytest.raises(ValidationError):
            Track(
                track_id=1,
                detection=sample_detection,
                state="confirmed",
                age=2,
                frames_since_update=5,
            )
    
    def test_invalid_state_raises(self, sample_detection):
        with pytest.raises(ValidationError):
            Track(
                track_id=1,
                detection=sample_detection,
                state="invalid_state",  # not in Literal
                age=0,
                frames_since_update=0,
            )


class TestFrameResult:
    def test_valid_with_segmented(self, sample_detection, sample_segmented):
        fr = FrameResult(
            frame_id=0,
            image_shape=(480, 640),
            detections=[sample_detection],
            segmented=[sample_segmented],
            tracks=[],
            prompts=["person"],
            inference_times={"detector": 100.0},
        )
        assert len(fr.detections) == len(fr.segmented)
    
    def test_segmented_length_mismatch_raises(self, sample_detection, sample_segmented):
        # 2 detections, 1 segmented — must fail
        with pytest.raises(ValidationError):
            FrameResult(
                frame_id=0,
                image_shape=(480, 640),
                detections=[sample_detection, sample_detection],
                segmented=[sample_segmented],
                tracks=[],
                prompts=["person"],
                inference_times={},
            )
    
    def test_empty_detections_valid(self, empty_frame_result):
        # Coming from fixture geliyor, validation passes
        assert len(empty_frame_result.detections) == 0
        assert empty_frame_result.segmented is None
    
    def test_segmented_none_when_no_detections(self, empty_frame_result):
        # Edge case: empty detections but segmented None — valid
        assert empty_frame_result.segmented is None