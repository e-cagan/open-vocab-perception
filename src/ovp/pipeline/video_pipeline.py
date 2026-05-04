"""
Module for video pipeline.
"""

from collections.abc import Iterator

import numpy as np

from ovp.core.interfaces import BaseDetector, BaseSegmenter, BaseTracker
from ovp.core.types import FrameResult, SegmentedDetection, Track


class VideoPipeline:
    """
    Video pipeline class.
    """

    def __init__(
        self,
        detector: BaseDetector,
        segmenter: BaseSegmenter | None = None,
        tracker: BaseTracker | None = None,
        keyframe_interval: int = 10,
    ) -> None:
        self._detector = detector
        self._segmenter = segmenter
        self._tracker = tracker
        self._keyframe_interval = keyframe_interval

    def run_video(
        self,
        frames: Iterator[np.ndarray],
        prompts: list[str],
        detector_threshold: float | None = None,
    ) -> Iterator[tuple[np.ndarray, FrameResult]]:
        """Process video frame stream. Yields FrameResult per frame."""
        # Edge case: empty prompt
        if not prompts:
            raise ValueError("prompts cannot be empty")

        # Reset the tracker if any (new video is starting)
        if self._tracker is not None:
            self._tracker.reset()

        # Cache: last keyframe result
        last_keyframe_result: FrameResult | None = None

        for frame_idx, frame in enumerate(frames):
            is_keyframe = frame_idx % self._keyframe_interval == 0

            if is_keyframe:
                # Detector
                detections = self._detector.detect(frame, prompts, threshold=detector_threshold)

                # Segmenter
                segmented = None
                if self._segmenter is not None and len(detections) > 0:
                    boxes = [d.bbox for d in detections]
                    masks = self._segmenter.segment(frame, boxes)
                    segmented = [
                        SegmentedDetection(detection=d, mask=m)
                        for d, m in zip(detections, masks, strict=True)
                    ]

                # Tracker
                tracks = []
                if self._tracker is not None:
                    tracks = self._tracker.update(detections)
                    if segmented is not None and len(tracks) > 0:
                        tracks = self._attach_masks_to_tracks(tracks, segmented)

                # Build result
                result = FrameResult(
                    frame_id=frame_idx,
                    image_shape=frame.shape[:2],
                    detections=detections,
                    segmented=segmented,
                    tracks=tracks,
                    prompts=prompts,
                    inference_times={},
                )
                last_keyframe_result = result
                yield frame, result

            else:
                # No keyframe
                if last_keyframe_result is None:
                    empty_result = FrameResult(
                        frame_id=frame_idx,
                        image_shape=frame.shape[:2],
                        detections=[],
                        segmented=None,
                        tracks=[],
                        prompts=prompts,
                        inference_times={},
                    )
                    yield frame, empty_result
                else:
                    # Cache the result for reuse
                    cached_result = last_keyframe_result.model_copy(
                        update={
                            "frame_id": frame_idx,
                            "image_shape": frame.shape[:2],
                        }
                    )
                    yield frame, cached_result

    def _attach_masks_to_tracks(
        self, tracks: list[Track], segmented: list[SegmentedDetection]
    ) -> list[Track]:
        """Match tracks to segmented detections by bbox IoU, attach masks."""
        updated = []
        for track in tracks:
            # Find nearest segmented detection
            best_iou = 0.0
            best_mask = None
            for sd in segmented:
                iou = track.detection.bbox.iou(sd.detection.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_mask = sd.mask

            if best_iou > 0.5:
                updated.append(track.model_copy(update={"mask": best_mask}))
            else:
                updated.append(track)

        return updated
