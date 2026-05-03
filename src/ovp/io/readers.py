"""
Module for file readers.
"""

from pathlib import Path
from typing import Iterator
import cv2
import numpy as np


class VideoReader:
    """Read frames from a video file as RGB numpy arrays."""
    
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Video not found: {self._path}")
        
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self._path}")
        
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ret, frame_bgr = self._cap.read()
            if not ret:
                break
            # OpenCV BGR → RGB (model standard)
            yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    def close(self) -> None:
        self._cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()