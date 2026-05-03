"""
Module for datatype classes.
"""

import numpy as np
from typing import Literal, Any
from pydantic import BaseModel, Field, model_validator, ConfigDict, field_validator
from typing_extensions import Self


class BoundingBox(BaseModel):
    """
    Bounding box model.
    """
    
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)
    format: Literal["xyxy"] = "xyxy"

    def iou(self, other: "BoundingBox") -> float:
        """Compute Intersection-over-Union with another bounding box."""
        # Intersection coordinates
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        
        # No overlap
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        
        inter_area = (ix2 - ix1) * (iy2 - iy1)
        
        # Areas
        self_area = (self.x2 - self.x1) * (self.y2 - self.y1)
        other_area = (other.x2 - other.x1) * (other.y2 - other.y1)
        union_area = self_area + other_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    @model_validator(mode="after")
    def _validate_corner_order(self) -> Self:
        if self.x2 <= self.x1:
            raise ValueError(f"x2 ({self.x2}) must be > x1 ({self.x1})")
        if self.y2 <= self.y1:
            raise ValueError(f"y2 ({self.y2}) must be > y1 ({self.y1})")
        return self
    

class Detection(BaseModel):
    """
    Detection model.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    bbox: BoundingBox
    score: float = Field(ge=0.0, le=1.0)
    label: str
    label_id: int | None = None
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Mask(BaseModel):
    """
    Binary segmentation mask for a single object.
    Shape: (H, W), dtype: bool.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    data: np.ndarray
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    label: str | None = None
    
    @field_validator("data")
    @classmethod
    def _validate_mask_array(cls, v: np.ndarray) -> np.ndarray:
        if not isinstance(v, np.ndarray):
            raise TypeError(f"data must be np.ndarray, got {type(v).__name__}")
        if v.ndim != 2:
            raise ValueError(f"data must be 2D (H, W), got shape {v.shape}")
        if v.dtype != np.bool_:
            raise ValueError(f"data must be bool dtype, got {v.dtype}")
        if v.size == 0:
            raise ValueError("data must not be empty")
        return v
    

class SegmentedDetection(BaseModel):
    """
    A detection paired with its segmentation mask.
    Both must reference the same physical object in the same frame.
    """
    
    detection: Detection
    mask: Mask


class Track(BaseModel):
    """
    A detection observed across multiple frames with a persistent identity.
    """
    
    track_id: int = Field(ge=0)
    detection: Detection
    mask: Mask | None = None
    state: Literal["tentative", "confirmed", "lost"] = "tentative"
    age: int = Field(default=0, ge=0)
    frames_since_update: int = Field(default=0, ge=0)
    history: list[BoundingBox] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_age_consistency(self) -> Self:
        if self.frames_since_update > self.age:
            raise ValueError(
                f"frames_since_update ({self.frames_since_update}) "
                f"cannot exceed age ({self.age})"
            )
        return self
    

class FrameResult(BaseModel):
    """
    All perception outputs for a single frame.
    
    detections: always populated when detector ran.
    segmented: populated when segmenter also ran (1:1 with detections).
    tracks: populated when tracker ran (video pipeline only).
    """
    
    frame_id: int = Field(ge=0)
    timestamp: float | None = None
    image_shape: tuple[int, int]  # (H, W)
    
    detections: list[Detection] = Field(default_factory=list)
    segmented: list[SegmentedDetection] | None = None
    tracks: list[Track] = Field(default_factory=list)
    
    prompts: list[str] = Field(default_factory=list)
    inference_times: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_segmented_alignment(self) -> Self:
        if self.segmented is not None:
            if len(self.segmented) != len(self.detections):
                raise ValueError(
                    f"segmented length ({len(self.segmented)}) "
                    f"must match detections length ({len(self.detections)})"
                )
        return self