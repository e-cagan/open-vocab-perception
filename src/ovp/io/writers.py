"""
Module for file writers.
"""

from pathlib import Path

import cv2
import numpy as np


class VideoWriter:
    """Write annotated frames to a video file."""

    def __init__(
        self,
        path: str | Path,
        fps: float,
        width: int,
        height: int,
        fourcc: str = "mp4v",
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        codec = cv2.VideoWriter_fourcc(*fourcc)
        self._writer = cv2.VideoWriter(str(self._path), codec, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self._path}")

    def write(self, frame_rgb: np.ndarray) -> None:
        """Write a single RGB frame."""
        # RGB → BGR (for cv2)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self._writer.write(frame_bgr)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
