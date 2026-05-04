"""
Module for visually annotating the outputs.
"""

import numpy as np
import supervision as sv

from ovp.core.types import FrameResult


class FrameAnnotator:
    """
    Frame annotator class.
    """

    def __init__(
        self,
        box_thickness: int = 2,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        mask_opacity: float = 0.4,
    ) -> None:
        self._box_annotator = sv.BoxAnnotator(thickness=box_thickness)
        self._label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=text_thickness,
        )
        self._mask_annotator = sv.MaskAnnotator(opacity=mask_opacity)

    def _to_supervision(self, fr: FrameResult) -> sv.Detections:
        # Take the coordinates of bboxes and confidence
        xyxy = np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in fr.detections])
        confidence = np.array([d.score for d in fr.detections])

        # Class ID — deterministic int from label string
        unique_labels = list(set(d.label for d in fr.detections))
        label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
        class_id = np.array([label_to_id[d.label] for d in fr.detections])

        # Fill up the detections to return
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        # Add mask if there's any
        if fr.segmented is not None:
            masks = np.stack([sd.mask.data for sd in fr.segmented])
            detections.mask = masks

        return detections

    def annotate(self, image: np.ndarray, frame_result: FrameResult) -> np.ndarray:
        # Edge case: no detection
        if len(frame_result.detections) == 0:
            return image.copy()

        # Convert detections to supervision format
        detections_sv = self._to_supervision(frame_result)

        # Annotate the image using mask and bbox
        annotated = image.copy()
        if frame_result.segmented is not None:
            annotated = self._mask_annotator.annotate(annotated, detections_sv)
        annotated = self._box_annotator.annotate(annotated, detections_sv)

        # Build track_id lookup if tracks exist
        track_lookup = {}
        if frame_result.tracks:
            track_lookup = {id(t.detection): t.track_id for t in frame_result.tracks}

        # Build labels with optional track ID prefix
        labels = []
        for d in frame_result.detections:
            track_id = track_lookup.get(id(d))
            if track_id is not None:
                labels.append(f"#{track_id} {d.label} {d.score:.2f}")
            else:
                labels.append(f"{d.label} {d.score:.2f}")

        annotated = self._label_annotator.annotate(annotated, detections_sv, labels=labels)

        return annotated
