"""
Module for sam2 object segmentator.
"""

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

from ovp.core.interfaces import BaseSegmenter
from ovp.core.registry import SEGMENTER_REGISTRY
from ovp.core.types import BoundingBox, Mask


@SEGMENTER_REGISTRY.register("sam2")
class Sam2Segmenter(BaseSegmenter):
    """
    SAM2 segmentator class
    """

    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-tiny",
        device: str = "cuda",
        dtype: str = "fp32",
        multimask_output: bool = True,
        mask_selection: str = "best_iou",
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._multimask_output = multimask_output
        self._mask_selection = mask_selection

        # Data type conversion
        if dtype == "fp16":
            self._torch_dtype = torch.float16
        elif dtype == "fp32":
            self._torch_dtype = torch.float32
        else:
            raise ValueError("Data type should be 'fp16' or 'fp32'")

        # Model and processor
        self.model = Sam2Model.from_pretrained(model_id, dtype=self._torch_dtype).to(device)
        self.processor = Sam2Processor.from_pretrained(model_id)

    def segment(
        self,
        image: np.ndarray,
        boxes: list[BoundingBox],
    ) -> list[Mask]:
        # Edge case: empty bbox
        if len(boxes) == 0:
            return list()

        # Convert numpy array to PIL image
        image_pil = Image.fromarray(image).convert("RGB")

        # Take coordinates from boxes within the expected format of SAM2
        input_boxes = [[[box.x1, box.y1, box.x2, box.y2] for box in boxes]]

        # Inference
        inputs = self.processor(image_pil, input_boxes=input_boxes, return_tensors="pt").to(
            self._device
        )
        inputs["pixel_values"] = inputs["pixel_values"].to(self._torch_dtype)

        # No gradients and data type autocasting
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda",
                dtype=self._torch_dtype,
                enabled=(self._torch_dtype != torch.float32),
            ),
        ):
            outputs = self.model(**inputs)

        # Post process the masks
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            original_sizes=inputs["original_sizes"].cpu(),
            reshaped_input_sizes=inputs["reshaped_input_sizes"].cpu(),
        )

        # Sanity check
        n = len(boxes)
        assert outputs.iou_scores.shape[1] == n, (
            f"SAM2 returned {outputs.iou_scores.shape[1]} boxes for {n} input boxes"
        )

        # Multi-mask selection
        result_masks: list[Mask] = []
        for box_idx, box in enumerate(boxes):
            # Take the best IoU score
            best_idx = torch.argmax(outputs.iou_scores[0, box_idx]).item()
            best_score = outputs.iou_scores[0, box_idx, best_idx].item()

            # Take the corresponding mask
            mask_array = masks[0][box_idx, best_idx].cpu().numpy()

            # Defensive: convert mask dtype to bool dtype (it should already have to be bool)
            if mask_array.dtype != np.bool_:
                mask_array = mask_array.astype(np.bool_)

            # Add the mask to result_masks list
            result_masks.append(
                Mask(
                    data=mask_array,
                    score=float(best_score),
                    label=box.label
                    if hasattr(box, "label")
                    else None,  # BoundingBox doesn't hold labels so, None
                )
            )

        return result_masks

    @property
    def device(self) -> str:
        return self._device

    @property
    def name(self) -> str:
        return f"sam2:{self._model_id.split('/')[-1]}"
