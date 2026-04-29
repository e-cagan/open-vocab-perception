"""
Module for base interfaces.
"""

from abc import ABC, abstractmethod
from .types import Detection, Mask, BoundingBox, Track
import numpy as np


class BaseDetector(ABC):
    """
    Abstract base class for open-vocabulary object detectors.
    
    Concrete subclasses must implement detect().
    Optional lifecycle methods (warmup, to_device) have sensible defaults.
    """
    
    @abstractmethod
    def detect(
        self, 
        image: np.ndarray, 
        prompts: list[str],
    ) -> list[Detection]:
        """
        Run detection on a single image with given text prompts.
        
        Args:
            image: RGB image as np.ndarray, shape (H, W, 3), dtype uint8.
            prompts: List of text queries (e.g. ["person", "red car"]).
        
        Returns:
            List of Detection objects, ordered by descending confidence.
        """
        ...
    
    def warmup(self, image_shape: tuple[int, int] = (480, 640)) -> None:
        """Run a dummy inference to trigger lazy initialization (CUDA kernels, etc)."""
        dummy = np.zeros((*image_shape, 3), dtype=np.uint8)
        _ = self.detect(dummy, prompts=["dummy"])
    
    @property
    @abstractmethod
    def device(self) -> str:
        """Current device the model resides on (e.g. 'cuda:0', 'cpu')."""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier (e.g. 'grounding-dino-tiny')."""
        ...


class BaseSegmenter(ABC):
    """Abstract base for promptable segmentation models."""
    
    @abstractmethod
    def segment(
        self,
        image: np.ndarray,
        boxes: list[BoundingBox],
    ) -> list[Mask]:
        """
        Generate segmentation masks for given bounding box prompts.
        
        Args:
            image: RGB image as np.ndarray, shape (H, W, 3), dtype uint8.
            boxes: List of BoundingBox prompts (one per object).
        
        Returns:
            List of Mask objects, one per input box, in the same order.
            Length of returned list equals length of input boxes.
        """
        ...
    
    def warmup(self, image_shape: tuple[int, int] = (480, 640)) -> None:
        dummy_image = np.zeros((*image_shape, 3), dtype=np.uint8)
        dummy_box = BoundingBox(x1=0, y1=0, x2=image_shape[1]-1, y2=image_shape[0]-1)
        _ = self.segment(dummy_image, boxes=[dummy_box])
    
    @property
    @abstractmethod
    def device(self) -> str: ...
    
    @property
    @abstractmethod
    def name(self) -> str: ...


class BaseTracker(ABC):
    """
    Abstract base for multi-object trackers.
    
    Trackers are stateful: each update() call mutates internal state.
    Use reset() between independent video sequences.
    """
    
    @abstractmethod
    def update(
        self,
        detections: list[Detection],
        frame_id: int,
    ) -> list[Track]:
        """
        Associate detections with existing tracks, create new tracks, expire lost ones.
        
        Args:
            detections: Current frame's detections.
            frame_id: Monotonically increasing frame index.
        
        Returns:
            Active tracks (confirmed + tentative). Lost tracks may be excluded.
        """
        ...
    
    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state. Call between independent video sequences."""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str: ...