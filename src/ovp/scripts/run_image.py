"""
CLI entrypoint for running the open-vocabulary perception pipeline on a single image.
"""

from pathlib import Path
import typer
import numpy as np
from PIL import Image
from rich.console import Console

from ovp.core.registry import DETECTOR_REGISTRY, SEGMENTER_REGISTRY
from ovp.pipeline.image_pipeline import ImagePipeline
from ovp.viz.annotators import FrameAnnotator

# Trigger registry registration (import side effects)
import ovp.detectors.grounding_dino  # noqa: F401
import ovp.segmenters.sam2  # noqa: F401

app = typer.Typer(help="Run open-vocabulary perception on a single image.")
console = Console()


@app.command()
def main(
    image: Path = typer.Option(..., "--image", "-i", help="Input image path", exists=True),
    prompts: str = typer.Option(..., "--prompts", "-p", help="Comma-separated text prompts"),
    output: Path = typer.Option(Path("annotated.jpg"), "--output", "-o", help="Output image path"),
    detector: str = typer.Option("grounding_dino", "--detector", help="Detector registry key"),
    segmenter: str = typer.Option("sam2", "--segmenter", help="Segmenter registry key"),
    threshold: float = typer.Option(0.3, "--threshold", "-t", help="Detector confidence threshold"),
    device: str = typer.Option("cuda", "--device", help="cuda or cpu"),
    no_segmenter: bool = typer.Option(False, "--no-segmenter", help="Disable segmenter"),
    fp16: bool = typer.Option(False, "--fp16", help="Use fp16 inference (faster, lower VRAM)"),
) -> None:
    """Run perception on a single image and save annotated output."""
    
    # Parse prompts
    # split by comma, strip whitespace, filter empty
    prompts_list = [p.strip() for p in prompts.split(",") if p.strip()]
    dtype = "fp16" if fp16 else "fp32"
    
    console.print(f"[cyan]Prompts:[/cyan] {prompts_list}")
    
    # Load image
    pil_img = Image.open(image).convert("RGB")
    image_np = np.asarray(pil_img)
    console.print(f"[cyan]Image shape:[/cyan] {image_np.shape}")
    
    # Build detector via registry
    # DETECTOR_REGISTRY.create kullan, threshold ve device pas et
    det_instance = DETECTOR_REGISTRY.create(detector, device=device, threshold=threshold, dtype=dtype)
    
    # Build segmenter via registry (if any segmentation)
    seg_instance = None
    if not no_segmenter:
        seg_instance = SEGMENTER_REGISTRY.create(segmenter, device=device, dtype=dtype)
    
    # Build pipeline
    pipeline = ImagePipeline(detector=det_instance, segmenter=seg_instance)
    
    # Run
    console.print("[cyan]Running pipeline...[/cyan]")
    result = pipeline.run(image_np, prompts=prompts_list)
    
    # Print summary
    console.print(f"[green]Detections: {len(result.detections)}[/green]")
    for d in result.detections:
        console.print(f"  - {d.label}: score={d.score:.3f}")
    console.print(f"[cyan]Inference times:[/cyan] {result.inference_times}")
    
    # Annotate
    annotator = FrameAnnotator()
    annotated = annotator.annotate(image_np, result)
    
    # Save
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(annotated).save(output)
    console.print(f"[green]Saved to {output}[/green]")


if __name__ == "__main__":
    app()