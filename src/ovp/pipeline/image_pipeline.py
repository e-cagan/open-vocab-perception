"""
Module for image pipeline.
"""

import time
import numpy as np

from ovp.core.interfaces import BaseDetector, BaseSegmenter
from ovp.core.types import (
    Detection, Mask, SegmentedDetection, FrameResult
)


class ImagePipeline:
    """
    Image pipeline class
    """
    
    def __init__(
        self,
        detector: BaseDetector,
        segmenter: BaseSegmenter | None = None,
    ) -> None:
        self._detector = detector
        self._segmenter = segmenter

    def run(
        self,
        image: np.ndarray,
        prompts: list[str],
        detector_threshold: float | None = None,
    ) -> FrameResult:
        # Sanity check: shape validation
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape {image.shape}")
        
        # Edge case: empty prompt handling
        if len(prompts) == 0:
            raise ValueError("prompts cannot be empty")
        
        # Detector and latency measurement
        t0 = time.perf_counter()
        detections = self._detector.detect(image, prompts, threshold=detector_threshold)
        detector_time = (time.perf_counter() - t0) * 1000

        # Segmenter call (if any)
        segmented_results: list[SegmentedDetection] | None = None
        segmenter_time: float | None = None

        # If there's no detection, then there is no point to trying segmentation
        if self._segmenter is not None and len(detections) > 0:
            t0 = time.perf_counter()
            boxes = [d.bbox for d in detections]
            masks = self._segmenter.segment(image, boxes)
            segmenter_time = (time.perf_counter() - t0) * 1000
            
            # Combine into SegmentedDetection
            segmented_results = [
                SegmentedDetection(detection=det, mask=mask)
                for det, mask in zip(detections, masks)
            ]

        # Create FrameResult and return it after filling the fields
        inference_times = {"detector": detector_time}
        if segmenter_time is not None:
            inference_times["segmenter"] = segmenter_time

        return FrameResult(
            frame_id=0,                       # image pipeline singe frame
            image_shape=image.shape[:2],      # (H, W)
            detections=detections,
            segmented=segmented_results,
            prompts=prompts,
            inference_times=inference_times,
        )