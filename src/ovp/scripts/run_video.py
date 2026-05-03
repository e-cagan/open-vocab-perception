"""CLI entrypoint for running the video pipeline."""
from pathlib import Path
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ovp.core.registry import DETECTOR_REGISTRY, SEGMENTER_REGISTRY, TRACKER_REGISTRY
from ovp.pipeline.video_pipeline import VideoPipeline
from ovp.viz.annotators import FrameAnnotator
from ovp.io.readers import VideoReader
from ovp.io.writers import VideoWriter

# Trigger registration
import ovp.detectors.grounding_dino  # noqa: F401
import ovp.segmenters.sam2  # noqa: F401
import ovp.trackers.bytetrack  # noqa: F401

app = typer.Typer(help="Run open-vocabulary perception on a video.")
console = Console()


@app.command()
def main(
    video: Path = typer.Option(..., "--video", "-v", help="Input video path", exists=True),
    prompts: str = typer.Option(..., "--prompts", "-p", help="Comma-separated prompts"),
    output: Path = typer.Option(Path("annotated.mp4"), "--output", "-o", help="Output video path"),
    detector: str = typer.Option("grounding_dino", "--detector"),
    segmenter: str = typer.Option("sam2", "--segmenter"),
    tracker: str = typer.Option("bytetrack", "--tracker"),
    threshold: float = typer.Option(0.3, "--threshold", "-t"),
    keyframe_interval: int = typer.Option(10, "--keyframe-interval", "-k"),
    device: str = typer.Option("cuda", "--device"),
    no_segmenter: bool = typer.Option(False, "--no-segmenter"),
    no_tracker: bool = typer.Option(False, "--no-tracker"),
    max_frames: int = typer.Option(0, "--max-frames", help="Limit frames (0 = all)"),
) -> None:
    """Run perception on a video and save annotated output."""
    
    # Parse prompts
    prompts_list = [p.strip() for p in prompts.split(",") if p.strip()]
    console.print(f"[cyan]Prompts:[/cyan] {prompts_list}")
    
    # Build components
    det_instance = DETECTOR_REGISTRY.create(detector, device=device, threshold=threshold)
    
    seg_instance = None
    if not no_segmenter:
        seg_instance = SEGMENTER_REGISTRY.create(segmenter, device=device)
    
    track_instance = None
    if not no_tracker:
        track_instance = TRACKER_REGISTRY.create(tracker)
    
    # Build pipeline
    pipeline = VideoPipeline(
        detector=det_instance,
        segmenter=seg_instance,
        tracker=track_instance,
        keyframe_interval=keyframe_interval,
    )
    
    annotator = FrameAnnotator(mask_opacity=0.4)
    
    # Run video
    with VideoReader(video) as reader:
        console.print(f"[cyan]Video:[/cyan] {reader.width}x{reader.height} @ {reader.fps:.1f} FPS, {reader.frame_count} frames")
        
        n_frames_to_process = min(reader.frame_count, max_frames) if max_frames > 0 else reader.frame_count
        
        with VideoWriter(output, fps=reader.fps, width=reader.width, height=reader.height) as writer:
            
            # Frame iterator with limit
            def limited_frames():
                for i, frame in enumerate(reader):
                    if max_frames > 0 and i >= max_frames:
                        break
                    yield frame
            
            # Pipeline + annotator + write loop
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Processing video...", total=n_frames_to_process)
                
                for frame, result in pipeline.run_video(limited_frames(), prompts=prompts_list):
                    annotated = annotator.annotate(frame, result)
                    writer.write(annotated)
                    progress.update(task, advance=1)
    
    console.print(f"[green]Saved to {output}[/green]")


if __name__ == "__main__":
    app()