"""
Benchmark OVP pipeline on COCO val2017 subset.

Computes:
- Detection mAP@0.5 
- Segmentation mIoU
- Mean FPS

Usage:
    python -m ovp.scripts.benchmark --num-images 100 --output results.json
"""

import torch
from torchmetrics.detection import MeanAveragePrecision

from pathlib import Path
import json
from typing import Iterator

import typer
import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from pycocotools.coco import COCO

# Trigger registration
import ovp.detectors.grounding_dino  # noqa: F401
import ovp.segmenters.sam2  # noqa: F401
import ovp.trackers.bytetrack  # noqa: F401

from ovp.core.types import BoundingBox
from ovp.core.registry import DETECTOR_REGISTRY, SEGMENTER_REGISTRY
from ovp.pipeline.image_pipeline import ImagePipeline


app = typer.Typer(help="Benchmark OVP pipeline on COCO val2017.")
console = Console()


# ============================================================
# COCO Helpers
# ============================================================

def coco_bbox_to_xyxy(coco_bbox: list[float]) -> tuple[float, float, float, float]:
    """COCO bbox format [x, y, w, h] -> xyxy [x1, y1, x2, y2]."""
    x, y, w, h = coco_bbox
    return (x, y, x + w, y + h)


def get_image_classes(coco: COCO, img_id: int) -> list[str]:
    """Get unique class names present in an image's annotations."""
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    cat_ids = sorted(set(ann["category_id"] for ann in anns))
    return [coco.loadCats(cid)[0]["name"] for cid in cat_ids]


def get_ground_truth(coco: COCO, img_id: int) -> dict:
    """
    Extract ground truth for one image.
    
    Returns:
        {
            "boxes": list of (x1, y1, x2, y2),
            "labels": list of class names,
            "masks": list of binary numpy arrays,
        }
    """
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    
    boxes = []
    labels = []
    masks = []
    
    for ann in anns:
        boxes.append(coco_bbox_to_xyxy(ann["bbox"]))
        cat_name = coco.loadCats(ann["category_id"])[0]["name"]
        labels.append(cat_name)
        # Polygon/RLE → binary mask (H, W)
        masks.append(coco.annToMask(ann).astype(bool))
    
    return {
        "boxes": boxes,
        "labels": labels,
        "masks": masks,
    }


# ============================================================
# Main Command
# ============================================================

@app.command()
def main(
    coco_root: Path = typer.Option(
        Path("data/coco"),
        "--coco-root",
        help="COCO dataset root (must contain annotations/ and val2017/).",
    ),
    num_images: int = typer.Option(100, "--num-images", "-n", help="Number of images to evaluate"),
    output: Path = typer.Option(Path("benchmark_results.json"), "--output", "-o"),
    detector_threshold: float = typer.Option(0.3, "--threshold", "-t"),
    seed: int = typer.Option(42, "--seed", help="Random seed for image sampling"),
) -> None:
    """Run benchmark on COCO val2017."""
    
    # Validate dataset paths
    ann_file = coco_root / "annotations" / "instances_val2017.json"
    img_dir = coco_root / "val2017"
    
    if not ann_file.exists():
        console.print(f"[red]Annotations not found at {ann_file}[/red]")
        raise typer.Exit(1)
    if not img_dir.exists():
        console.print(f"[red]Images not found at {img_dir}[/red]")
        raise typer.Exit(1)
    
    # Load COCO
    console.print(f"[cyan]Loading COCO annotations...[/cyan]")
    coco = COCO(str(ann_file))
    all_image_ids = coco.getImgIds()
    console.print(f"Total val images: {len(all_image_ids)}")

    # Build COCO class name -> ID mapping (stable across images)
    cat_map_name_to_id = {}
    for cat_id in coco.getCatIds():
        cat_info = coco.loadCats(cat_id)[0]
        cat_map_name_to_id[cat_info["name"]] = cat_id
    console.print(f"COCO classes available: {len(cat_map_name_to_id)}")
    
    # Sample subset
    rng = np.random.default_rng(seed)
    selected_ids = rng.choice(all_image_ids, size=min(num_images, len(all_image_ids)), replace=False)
    selected_ids = sorted(selected_ids.tolist())
    console.print(f"Sampled {len(selected_ids)} images for benchmark")
    
    # Build pipeline
    console.print("[cyan]Building pipeline...[/cyan]")
    detector = DETECTOR_REGISTRY.create("grounding_dino", threshold=detector_threshold)
    segmenter = SEGMENTER_REGISTRY.create("sam2")
    pipeline = ImagePipeline(detector=detector, segmenter=segmenter)

    # Initialize metric accumulators
    metric_bbox = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=False,
    )
    
    # Run benchmark loop
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Benchmarking...", total=len(selected_ids))
        
        for img_id in selected_ids:
            # Get image info & ground truth
            img_info = coco.loadImgs(int(img_id))[0]
            img_path = img_dir / img_info["file_name"]
            
            classes_in_image = get_image_classes(coco, int(img_id))
            
            if not classes_in_image:
                # No annotations — skip
                progress.update(task, advance=1)
                continue
            
            # prompts = ground truth class names
            prompts = classes_in_image
            
            # Load image
            image_np = np.array(Image.open(img_path).convert("RGB"))
            
            # Run pipeline
            result = pipeline.run(image_np, prompts=prompts)

            # Get ground truth for this image
            gt = get_ground_truth(coco, int(img_id))

            # Build prediction tensors for torchmetrics
            pred_boxes = []
            pred_scores = []
            pred_labels = []
            for det in result.detections:
                if det.label not in cat_map_name_to_id:
                    continue
                pred_boxes.append([det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2])
                pred_scores.append(det.score)
                pred_labels.append(cat_map_name_to_id[det.label])

            if pred_boxes:
                preds = [{
                    "boxes": torch.tensor(pred_boxes, dtype=torch.float32),
                    "scores": torch.tensor(pred_scores, dtype=torch.float32),
                    "labels": torch.tensor(pred_labels, dtype=torch.int64),
                }]
            else:
                preds = [{
                    "boxes": torch.empty((0, 4), dtype=torch.float32),
                    "scores": torch.empty((0,), dtype=torch.float32),
                    "labels": torch.empty((0,), dtype=torch.int64),
                }]

            # Build ground truth tensors
            gt_boxes = []
            gt_labels = []
            for box, label in zip(gt["boxes"], gt["labels"]):
                if label not in cat_map_name_to_id:
                    continue
                gt_boxes.append(list(box))
                gt_labels.append(cat_map_name_to_id[label])

            if gt_boxes:
                targets = [{
                    "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
                    "labels": torch.tensor(gt_labels, dtype=torch.int64),
                }]
            else:
                targets = [{
                    "boxes": torch.empty((0, 4), dtype=torch.float32),
                    "labels": torch.empty((0,), dtype=torch.int64),
                }]

            # Update metric (accumulates across all images)
            metric_bbox.update(preds, targets)

            # Record raw stats per image (for JSON)
            results.append({
                "image_id": int(img_id),
                "filename": img_info["file_name"],
                "prompts": prompts,
                "n_detections": len(result.detections),
                "n_ground_truth": len(gt["boxes"]),
                "detector_ms": result.inference_times.get("detector", 0),
                "segmenter_ms": result.inference_times.get("segmenter", 0),
            })

            progress.update(task, advance=1)
    
    # Compute final metrics
    console.print("\n[cyan]Computing detection mAP...[/cyan]")
    bbox_metrics = metric_bbox.compute()
    map_50 = float(bbox_metrics["map_50"])
    map_overall = float(bbox_metrics["map"])
    map_75 = float(bbox_metrics["map_75"])

    # Compute latency averages
    avg_detections = float(np.mean([r["n_detections"] for r in results]))
    avg_detector_ms = float(np.mean([r["detector_ms"] for r in results]))
    avg_segmenter_ms = float(np.mean([r["segmenter_ms"] for r in results]))
    effective_fps = 1000 / (avg_detector_ms + avg_segmenter_ms)

    # Save full results
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({
            "config": {
                "num_images": len(selected_ids),
                "detector_threshold": detector_threshold,
            },
            "metrics": {
                "mAP_50": map_50,
                "mAP_75": map_75,
                "mAP_50_95": map_overall,
                "avg_detections_per_image": avg_detections,
                "avg_detector_latency_ms": avg_detector_ms,
                "avg_segmenter_latency_ms": avg_segmenter_ms,
                "effective_fps": effective_fps,
            },
            "per_image": results,
        }, f, indent=2)

    # Print summary
    console.print(f"[green]Saved results to {output}[/green]")
    console.print(f"\n[cyan]Detection accuracy:[/cyan]")
    console.print(f"  mAP@0.5:      {map_50:.4f}")
    console.print(f"  mAP@0.75:     {map_75:.4f}")
    console.print(f"  mAP@0.5:0.95: {map_overall:.4f}")
    console.print(f"\n[cyan]Latency:[/cyan]")
    console.print(f"  Avg detections per image: {avg_detections:.1f}")
    console.print(f"  Avg detector latency:     {avg_detector_ms:.1f} ms")
    console.print(f"  Avg segmenter latency:    {avg_segmenter_ms:.1f} ms")
    console.print(f"  Effective FPS:            {effective_fps:.2f}")


if __name__ == "__main__":
    app()