"""Tests for ByteTracker — the supervision-backed tracker wrapper."""

from ovp.core.registry import TRACKER_REGISTRY
from ovp.core.types import BoundingBox, Detection, Track
from ovp.trackers.bytetrack import ByteTracker

# ============================================================
# Helpers
# ============================================================


def make_detection(x1, y1, x2, y2, score=0.85, label="person"):
    """Helper to construct a Detection from bbox coordinates."""
    return Detection(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        score=score,
        label=label,
    )


# ============================================================
# Basic API & Registry
# ============================================================


class TestByteTrackerBasics:
    def test_registry_contains_bytetrack(self):
        assert "bytetrack" in TRACKER_REGISTRY

    def test_default_construction(self):
        """Tracker with default params should construct cleanly."""
        tracker = ByteTracker()
        assert tracker.name == "bytetrack"

    def test_custom_construction(self):
        """Tracker should accept custom parameters."""
        tracker = ByteTracker(
            track_activation_threshold=0.5,
            lost_track_buffer=60,
            minimum_matching_threshold=0.7,
            frame_rate=60,
            minimum_consecutive_frames=3,
        )
        assert tracker.name == "bytetrack"

    def test_create_via_registry(self):
        """Registry create() should produce a working tracker."""
        tracker = TRACKER_REGISTRY.create("bytetrack")
        assert isinstance(tracker, ByteTracker)


# ============================================================
# Tracking Behavior
# ============================================================


class TestSingleObjectTracking:
    def test_single_object_consistent_id(self):
        """A single object moving slightly should keep the same track_id."""
        tracker = ByteTracker()
        track_ids_seen = []

        for frame_idx in range(5):
            x_offset = frame_idx * 10
            detections = [make_detection(100 + x_offset, 100, 200 + x_offset, 300)]
            tracks = tracker.update(detections)

            assert len(tracks) == 1, f"Frame {frame_idx}: expected 1 track"
            track_ids_seen.append(tracks[0].track_id)

        # Tüm frame'lerde aynı track_id olmalı
        assert len(set(track_ids_seen)) == 1, f"Track ID changed mid-sequence: {track_ids_seen}"

    def test_track_returns_correct_type(self):
        """update() should return list of Track objects."""
        tracker = ByteTracker()
        detections = [make_detection(100, 100, 200, 300)]
        tracks = tracker.update(detections)

        assert all(isinstance(t, Track) for t in tracks)

    def test_track_preserves_detection_label(self):
        """Track should carry the original detection's label."""
        tracker = ByteTracker()
        detections = [make_detection(100, 100, 200, 300, label="cat")]
        tracks = tracker.update(detections)

        assert len(tracks) == 1
        assert tracks[0].detection.label == "cat"


class TestMultiObjectTracking:
    def test_two_objects_distinct_ids(self):
        """Two well-separated objects should get distinct track_ids."""
        tracker = ByteTracker()

        detections = [
            make_detection(100, 100, 200, 300),  # left
            make_detection(400, 100, 500, 300),  # right
        ]
        tracks = tracker.update(detections)

        assert len(tracks) == 2
        ids = {t.track_id for t in tracks}
        assert len(ids) == 2, "Two objects should get two distinct IDs"

    def test_two_objects_consistent_ids_over_time(self):
        """Two objects tracked over multiple frames keep their IDs."""
        tracker = ByteTracker()
        first_frame_ids = None

        for frame_idx in range(3):
            detections = [
                make_detection(100, 100, 200, 300),
                make_detection(400, 100, 500, 300),
            ]
            tracks = tracker.update(detections)
            assert len(tracks) == 2

            current_ids = sorted([t.track_id for t in tracks])
            if first_frame_ids is None:
                first_frame_ids = current_ids
            else:
                assert current_ids == first_frame_ids, f"Frame {frame_idx}: track IDs changed"


# ============================================================
# State Management
# ============================================================


class TestResetBehavior:
    def test_reset_clears_state(self):
        """After reset, new objects start fresh."""
        tracker = ByteTracker()

        # First video
        tracker.update([make_detection(100, 100, 200, 300)])
        tracker.update([make_detection(110, 100, 210, 300)])

        # Reset
        tracker.reset()

        # New video — should start fresh
        tracks = tracker.update([make_detection(50, 50, 150, 250, label="dog")])

        # ID counter ya 1'den başlar ya da düşük başlar — önemli olan tracker hâlâ çalışıyor
        assert len(tracks) == 1

    def test_reset_returns_id_to_one(self):
        """After reset, track IDs should restart from 1."""
        tracker = ByteTracker()

        # Bazı objeler track et
        for _ in range(3):
            tracker.update([make_detection(100, 100, 200, 300)])

        # Reset
        tracker.reset()

        # Yeni obje — ID=1 bekleniyor
        tracks = tracker.update([make_detection(50, 50, 150, 250)])

        assert len(tracks) == 1
        assert tracks[0].track_id == 1


# ============================================================
# Edge Cases
# ============================================================


class TestEdgeCases:
    def test_empty_detections_returns_empty(self):
        """Empty input should not crash and return empty tracks."""
        tracker = ByteTracker()
        tracks = tracker.update([])

        assert tracks == []

    def test_empty_first_frame_then_detections_no_tracks(self):
        """ByteTrack does not create new tracks if the first frame was empty."""
        tracker = ByteTracker()

        tracker.update([])  # boş ilk frame
        tracks = tracker.update([make_detection(100, 100, 200, 300)])

        # ByteTrack semantiği: ilk frame boşsa sonraki frame'de track açmıyor
        assert tracks == []

    def test_low_confidence_detection_handled(self):
        """Low-confidence detections shouldn't crash the tracker."""
        tracker = ByteTracker()
        # Threshold'un altında bir detection
        tracks = tracker.update([make_detection(100, 100, 200, 300, score=0.1)])

        # ByteTrack low-confidence detection'ları filtreleyebilir
        # Önemli olan crash etmemesi
        assert isinstance(tracks, list)
