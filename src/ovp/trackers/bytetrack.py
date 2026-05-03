"""
Module for ByteTrack tracker.
"""

from typing import Optional
import numpy as np
import supervision as sv

from ovp.core.interfaces import BaseTracker
from ovp.core.types import Detection, Track, BoundingBox
from ovp.core.registry import TRACKER_REGISTRY


@TRACKER_REGISTRY.register("bytetrack")
class ByteTracker(BaseTracker):
    """
    Wraps supervision.ByteTrack for cross-frame identity assignment.
    
    Stateful: maintains track state between calls. Call reset() to clear
    state when starting a new video.
    """
    
    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        minimum_consecutive_frames: int = 1,
    ) -> None:
        self._track_thresh = track_activation_threshold
        self._lost_buffer = lost_track_buffer
        self._match_thresh = minimum_matching_threshold
        self._frame_rate = frame_rate
        self._min_consec = minimum_consecutive_frames
        
        # Internal supervision tracker
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )
    
    @property
    def name(self) -> str:
        return "bytetrack"
    
    def update(self, detections: list[Detection]) -> list[Track]:
        """
        Process detections from one frame, return tracks with persistent IDs.
        """
        # Edge case: no detections
        if len(detections) == 0:
            sv_detections = sv.Detections.empty()
            self._tracker.update_with_detections(sv_detections)
            # No new tracks to return; lost tracks are managed internally
            return []
        
        # Convert Detection list -> sv.Detections
        sv_detections = self._to_supervision(detections)
        
        # ByteTrack update — input + state -> tracks with tracker_id
        tracked_sv = self._tracker.update_with_detections(sv_detections)
        
        # Convert back to Track list
        return self._to_tracks(tracked_sv, detections)
    
    def reset(self) -> None:
        """Reset internal tracker state. Call before processing a new video."""
        self._tracker = sv.ByteTrack(
            track_activation_threshold=self._track_thresh,
            lost_track_buffer=self._lost_buffer,
            minimum_matching_threshold=self._match_thresh,
            frame_rate=self._frame_rate,
            minimum_consecutive_frames=self._min_consec,
        )
    
    def _to_supervision(self, detections: list[Detection]) -> sv.Detections:
        """Convert list[Detection] to sv.Detections format."""
        xyxy = np.array([
            [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2]
            for d in detections
        ])
        confidence = np.array([d.score for d in detections])
        
        unique_labels = sorted(set(d.label for d in detections))
        label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
        class_id = np.array([label_to_id[d.label] for d in detections])
        
        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )
    
    def _to_tracks(
        self, 
        tracked_sv: sv.Detections, 
        original_detections: list[Detection]
    ) -> list[Track]:
        """Convert sv.Detections (with tracker_id) back to list[Track]."""
        if tracked_sv.tracker_id is None or len(tracked_sv) == 0:
            return []
        
        tracks = []
        for i in range(len(tracked_sv)):
            tracker_id = int(tracked_sv.tracker_id[i])
            tracked_bbox = tracked_sv.xyxy[i]
            
            # Match to original detection by IoU
            match_idx = self._match_to_original(tracked_bbox, original_detections)
            if match_idx is None:
                continue
            
            track = Track(
                track_id=tracker_id,
                detection=original_detections[match_idx],
                mask=None,
                state="confirmed",
                age=0,
                frames_since_update=0,
                history=[],
            )
            tracks.append(track)
        
        return tracks
    
    @staticmethod
    def _match_to_original(
        tracked_bbox: np.ndarray, 
        detections: list[Detection],
        iou_threshold: float = 0.95,
    ) -> Optional[int]:
        """Find the index of the original detection matching this bbox via IoU."""
        # Convert numpy bbox to BoundingBox for iou() method
        tracked = BoundingBox(
            x1=float(tracked_bbox[0]),
            y1=float(tracked_bbox[1]),
            x2=float(tracked_bbox[2]),
            y2=float(tracked_bbox[3]),
        )
        
        best_iou = 0.0
        best_idx = None
        
        for i, det in enumerate(detections):
            iou = tracked.iou(det.bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        
        return best_idx if best_iou >= iou_threshold else None