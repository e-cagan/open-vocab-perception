"""Test ByteTracker with synthetic detection sequence."""
import numpy as np
from ovp.core.types import Detection, BoundingBox
from ovp.core.registry import TRACKER_REGISTRY
from ovp.trackers.bytetrack import ByteTracker  # registry trigger

# 1) Registry kontrolü
print(f"Registered trackers: {TRACKER_REGISTRY.keys()}")
assert "bytetrack" in TRACKER_REGISTRY

# 2) Tracker yarat
tracker = ByteTracker()
print(f"Tracker name: {tracker.name}")

# 3) Synthetic test — bir kişi 5 frame boyunca yürüyor (sağa kayıyor)
print("\n--- Test 1: Single moving object ---")
for frame_idx in range(5):
    # Her frame'de bbox biraz sağa kayıyor
    x_offset = frame_idx * 10
    detections = [
        Detection(
            bbox=BoundingBox(x1=100 + x_offset, y1=100, x2=200 + x_offset, y2=300),
            score=0.85,
            label="person",
        )
    ]
    tracks = tracker.update(detections)
    print(f"Frame {frame_idx}: {len(tracks)} tracks")
    for t in tracks:
        print(f"  track_id={t.track_id}, label={t.detection.label}, "
              f"bbox=({t.detection.bbox.x1:.0f}, {t.detection.bbox.y1:.0f}, "
              f"{t.detection.bbox.x2:.0f}, {t.detection.bbox.y2:.0f})")

# Aynı track_id olmalı tüm frame'lerde
# Track tutarlılığı kontrol — supervision.ByteTrack minimum_consecutive_frames=1 default ile
# ilk frame'de bile track_id atar

# 4) Reset testi
print("\n--- Test 2: Reset & new video ---")
tracker.reset()

# Yeni video, farklı obje
detections = [
    Detection(
        bbox=BoundingBox(x1=300, y1=300, x2=400, y2=500),
        score=0.90,
        label="car",
    )
]
tracks = tracker.update(detections)
print(f"After reset, first frame: {len(tracks)} tracks")
if tracks:
    print(f"  New track_id: {tracks[0].track_id}")
    # Reset sonrası track ID'ler de sıfırlanmış olmalı (1'den başlamalı)

# 5) Multi-object testi
print("\n--- Test 3: Multi-object tracking ---")
tracker.reset()

for frame_idx in range(3):
    detections = [
        Detection(
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=300),
            score=0.85,
            label="person",
        ),
        Detection(
            bbox=BoundingBox(x1=400, y1=100, x2=500, y2=300),
            score=0.80,
            label="person",
        ),
    ]
    tracks = tracker.update(detections)
    print(f"Frame {frame_idx}: {len(tracks)} tracks: " + 
          ", ".join(f"id={t.track_id}" for t in tracks))

# 6) Empty detection edge case
print("\n--- Test 4: Empty detection frame ---")
empty_tracks = tracker.update([])
print(f"Empty input: {len(empty_tracks)} tracks returned")

print("\nAll tests completed.")