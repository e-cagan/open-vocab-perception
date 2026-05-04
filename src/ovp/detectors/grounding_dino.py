"""
Module for grounding dino detector.
"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from ovp.core.interfaces import BaseDetector
from ovp.core.registry import DETECTOR_REGISTRY
from ovp.core.types import BoundingBox, Detection


@DETECTOR_REGISTRY.register("grounding_dino")
class GroundingDinoDetector(BaseDetector):
    """
    Grounding DINO detector class.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cuda",
        dtype: str = "fp16",
        threshold: float = 0.3,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._threshold = threshold

        # Data type check
        if dtype == "fp16":
            self._torch_dtype = torch.float16
        elif dtype == "fp32":
            self._torch_dtype = torch.float32
        else:
            raise ValueError("Data type should be 'fp16' or 'fp32'")

        # Model and processor
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, dtype=self._torch_dtype
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def detect(
        self,
        image: np.ndarray,
        prompts: list[str],
        threshold: float | None = None,
    ) -> list[Detection]:
        threshold = threshold if threshold is not None else self._threshold

        # Preprocess the text and image
        text_for_processor = ". ".join(prompts) + "."  # ["cat", "remote"] -> "cat. remote."
        text_labels_for_post = [text_for_processor]  # double bracket
        image_pil = Image.fromarray(image).convert("RGB")

        # Process tensors and cast pixel_values to model's dtype to avoid float/half mismatch
        inputs = self.processor(images=image_pil, text=text_for_processor, return_tensors="pt").to(
            self._device
        )
        inputs["pixel_values"] = inputs["pixel_values"].to(self._torch_dtype)

        # Outputs without gradients autocast to optimize
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda",
                dtype=self._torch_dtype,
                enabled=(self._torch_dtype != torch.float32),
            ),
        ):
            outputs = self.model(**inputs)

        # Take the best result
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=[image.shape[:2]],  # image numpy → shape = (H, W, 3) → [:2] = (H, W)
            text_labels=text_labels_for_post,
        )
        result = results[0]

        # Convert result to detections
        detections: list[Detection] = []
        boxes = result["boxes"].cpu().numpy()
        scores = result["scores"].cpu().numpy()
        text_labels = result["text_labels"]
        H, W = image.shape[:2]

        # Sanity check before iteration
        n = len(boxes)
        assert len(scores) == n and len(text_labels) == n, "HF output size mismatch"

        for box, score, label in zip(boxes, scores, text_labels, strict=True):
            x1, y1, x2, y2 = box.tolist()

            # Clip to frame bounds (model can produce slightly out-of-frame coords)
            x1 = max(0.0, min(x1, float(W)))
            y1 = max(0.0, min(y1, float(H)))
            x2 = max(0.0, min(x2, float(W)))
            y2 = max(0.0, min(y2, float(H)))

            detections.append(
                Detection(
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    score=float(score),
                    label=label,
                )
            )

        return detections

    @property
    def device(self) -> str:
        return self._device

    @property
    def name(self) -> str:
        return f"grounding-dino:{self._model_id.split('/')[-1]}"
