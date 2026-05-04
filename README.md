# Open-Vocabulary Perception (OVP)

> A modular pipeline for **text-prompted object detection, segmentation, and tracking**, combining GroundingDINO (open-vocabulary detection) with SAM 2 (class-agnostic segmentation) and ByteTrack (cross-frame identity). Built with a strict no-vibe-coding philosophy: every component is hand-written, every design decision documented.

![Hero](outputs/demo_video.gif)

*Real-time tracking on a busy İstanbul pedestrian scene. Track IDs (#1, #2, ...) remain consistent across frames; bounding boxes and masks update on keyframes via GroundingDINO + SAM 2.*

```bash
ovp-video -v input.mp4 -p "person" -o annotated.mp4
```

---

## Why This Project

Open-vocabulary perception is the natural evolution of closed-class detection. Instead of training on a fixed taxonomy, the model accepts arbitrary text queries at inference time. This project demonstrates how to compose three state-of-the-art components — **GroundingDINO** (text → bounding box), **SAM 2** (bounding box → pixel mask), and **ByteTrack** (cross-frame identity) — into a unified, production-ready pipeline that scales from single images to streaming video.

Built and tested on RTX 3050 Ti Laptop (4GB VRAM). All three components fit comfortably (~530MB total VRAM when loaded together).

---

## Architecture

```
text prompts ──► GroundingDinoDetector ──► Detection (bbox + label + score)
                                                 │
                            ┌────────────────────┤
                            ▼                    ▼
                    Sam2Segmenter         ByteTracker
                            │                    │
                            ▼                    ▼
                          Mask              Track (persistent ID)
                            │                    │
                            └────────┬───────────┘
                                     ▼
                    ImagePipeline / VideoPipeline
                                     │
                                     ▼
                              FrameResult (Pydantic)
```

Strategy pattern + Dependency Injection throughout. Each component implements an abstract base class (`BaseDetector`, `BaseSegmenter`, `BaseTracker`) and registers itself in a type-safe registry. Adding a new detector means writing one file — no changes to pipeline code.

```
src/ovp/
├── core/             # types, interfaces, registry
├── detectors/        # GroundingDinoDetector
├── segmenters/       # Sam2Segmenter
├── trackers/         # ByteTracker
├── pipeline/         # ImagePipeline, VideoPipeline
├── viz/              # FrameAnnotator (track-aware)
├── io/               # VideoReader, VideoWriter
└── scripts/          # ovp-image, ovp-video, ovp-benchmark CLI entry points
```

---

## Installation

Requires Python 3.10+ and a CUDA-capable GPU (tested on CUDA 13.0).

```bash
# Clone and enter
git clone https://github.com/e-cagan/open-vocab-perception.git
cd open-vocab-perception

# Create environment
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# PyTorch (match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install package + dependencies
pip install -e ".[notebook]"
```

---

## Usage

### Image Pipeline

Detect and segment a single image:

```bash
ovp-image -i path/to/image.jpg -p "person,car,bicycle" -o annotated.jpg
```

Detection only (skip segmentation for faster inference):

```bash
ovp-image -i path/to/image.jpg -p "cat" --no-segmenter -o detection_only.jpg
```

Tune confidence threshold:

```bash
ovp-image -i path/to/image.jpg -p "person" -t 0.5 -o filtered.jpg
```

See `ovp-image --help` for all options.

### Video Pipeline

Run the full pipeline on a video with persistent tracking:

```bash
ovp-video \
  --video input.mp4 \
  --prompts "person,car" \
  --output annotated.mp4 \
  --keyframe-interval 10
```

The `--keyframe-interval` flag controls how often the detector and segmenter run (default: every 10 frames). Between keyframes, ByteTrack maintains object identity, and the cached keyframe results are reused — this is the key trade-off for video latency.

Optional flags: `--no-tracker`, `--no-segmenter`, `--max-frames N` (limit frames for testing).

See `ovp-video --help` for all options.

### Programmatic API

Image:

```python
from ovp.detectors.grounding_dino import GroundingDinoDetector
from ovp.segmenters.sam2 import Sam2Segmenter
from ovp.pipeline.image_pipeline import ImagePipeline
from ovp.viz.annotators import FrameAnnotator
import numpy as np
from PIL import Image

detector = GroundingDinoDetector(threshold=0.3)
segmenter = Sam2Segmenter()
pipeline = ImagePipeline(detector=detector, segmenter=segmenter)

image = np.array(Image.open("scene.jpg").convert("RGB"))
result = pipeline.run(image, prompts=["person", "car"])

print(f"Found {len(result.detections)} objects")
print(f"Latency: {result.inference_times}")

annotated = FrameAnnotator().annotate(image, result)
Image.fromarray(annotated).save("output.jpg")
```

Video:

```python
from ovp.detectors.grounding_dino import GroundingDinoDetector
from ovp.segmenters.sam2 import Sam2Segmenter
from ovp.trackers.bytetrack import ByteTracker
from ovp.pipeline.video_pipeline import VideoPipeline
from ovp.viz.annotators import FrameAnnotator
from ovp.io.readers import VideoReader
from ovp.io.writers import VideoWriter

pipeline = VideoPipeline(
    detector=GroundingDinoDetector(threshold=0.3),
    segmenter=Sam2Segmenter(),
    tracker=ByteTracker(),
    keyframe_interval=10,
)
annotator = FrameAnnotator()

with VideoReader("input.mp4") as reader, \
     VideoWriter("output.mp4", fps=reader.fps, width=reader.width, height=reader.height) as writer:
    for frame, result in pipeline.run_video(iter(reader), prompts=["person"]):
        annotated = annotator.annotate(frame, result)
        writer.write(annotated)
```

---

## Examples

### Detection + Segmentation vs Detection Only

The pipeline supports detection-only mode for use cases where masks are not needed (e.g., bbox tracking, lightweight inference).

| Full pipeline | Detection only |
|---|---|
| ![Full](outputs/demo_01_hero_cats.jpg) | ![Det only](outputs/demo_02_detection_only_cats.jpg) |

### Threshold Tuning

The detector confidence threshold controls precision/recall trade-off. Default 0.3 works well for most scenes; tune up for cleaner output, down for higher recall.

| `-t 0.2` (low) | `-t 0.5` (high) |
|---|---|
| ![Low](outputs/demo_06a_thresh_low.jpg) | ![High](outputs/demo_06b_thresh_high.jpg) |

Low threshold produces many low-confidence false positives. High threshold may miss valid objects entirely. The default of 0.3 sits in the practical sweet spot.

---

## Performance

### Quantitative Benchmark on COCO val2017

Evaluated on a random 100-image sample from COCO val2017 (seed=42), RTX 3050 Ti Laptop GPU, fp32, detector threshold=0.3:

| Metric | Value |
|---|---|
| **Detection mAP@0.5** | 0.74 |
| **Detection mAP@0.75** | 0.60 |
| **Detection mAP@[0.5:0.95]** | 0.57 |
| **Segmentation mIoU** | 0.77 ± 0.18 |
| **Avg latency per image** | ~552 ms |
| **Effective FPS** | ~1.8 |

#### Methodology

Prompts are built **per-image from the ground-truth class labels** present in that image. For example, an image containing person and bicycle is prompted with `["person", "bicycle"]`. This represents the **informed user** use case — when the operator knows what classes they are looking for. It is an optimistic benchmark; results would be lower under blind prompting where all 80 COCO classes are queried simultaneously.

mIoU is computed over the subset of detections that match a ground-truth instance at bbox IoU≥0.5 (519 of 861 detections in the test sample). Mask IoU is then computed pixel-wise between the predicted mask and the matched ground-truth mask.

Reproduce with:

```bash
ovp-benchmark --num-images 100 --seed 42 --output benchmark_results.json
```

### Component Latency (Sandbox)

Single-image inference, isolated component timing on a 640×480 image:

| Component | Latency | VRAM |
|---|---|---|
| GroundingDINO-tiny (detector) | ~430 ms | 337 MB |
| SAM 2.1 Hiera-tiny (segmenter, 4 boxes) | ~160 ms | 121 MB |
| ByteTrack (tracker, per frame) | <5 ms | negligible |
| **Image pipeline (end-to-end)** | **~600 ms** | **~526 MB** |
| **Video pipeline (avg per frame, k=10)** | **~65 ms** | **~530 MB** |

SAM 2's promptable architecture decouples image encoding (~140 ms, runs once) from mask decoding (~4 ms per box, scales with prompt count). Dense scenes with 10–20 detections do not significantly slow segmentation.

The video pipeline exploits this with a keyframe strategy: detector + segmenter run on every Nth frame, ByteTrack maintains identity in between. Effective throughput is ~15 FPS at `keyframe_interval=10`.

---

## Limitations

This project is honest about what works and what doesn't.

### Compound queries are weakly enforced

GroundingDINO treats attributes as soft hints, not hard constraints. Querying for `"person wearing white shirt"` will likely match all persons regardless of clothing.

![Compound](outputs/demo_04_compound_queries.jpg)

For attribute-sensitive use cases, consider noun-only prompts followed by a downstream classifier (e.g., CLIP-based filtering on detection crops).

### Crowded scenes lack post-processing

The pipeline currently lacks proper NMS or duplicate suppression. In dense scenes, multiple overlapping bounding boxes for the same person can produce visually noisy mask overlays. This is reflected in the benchmark: 342 of 861 detections (≈40%) did not match a ground-truth instance, partly due to duplicate detections counted as false positives.

![Crowded](outputs/demo_03_crowded_scene.jpg)

A future version will integrate class-aware NMS in the pipeline orchestrator.

### Bounding boxes are static between keyframes

`VideoPipeline` caches the last keyframe's results between detector calls. Track IDs are preserved by ByteTrack, but bbox coordinates do not update between keyframes — a moving person's box stays in the keyframe position for up to N-1 frames. Visible at low keyframe intervals; reduces with larger keyframe gaps. Future versions will use SAM 2's video predictor for inter-keyframe mask propagation.

### Latency is not real-time on consumer GPUs

Per-image inference is ~550-600 ms on RTX 3050 Ti (fp32), giving ~1.8 FPS for image mode and ~15 FPS effective for video mode (with `keyframe_interval=10`). Real-time (30 FPS) requires either a more powerful GPU, larger keyframe intervals, or fp16 inference (planned).

### Mask quality varies with object size

Segmentation mIoU is 0.77 ± 0.18, with the high standard deviation reflecting that SAM 2 produces excellent masks for large, well-defined objects (often >0.90 IoU) but can struggle with small or thin objects (sometimes <0.50 IoU). The benchmark sample mIoU is dominated by the larger objects in each scene.

---

## Sandbox Notebooks

Each model was systematically tested before being wrapped in production classes. The notebooks document threshold sensitivity, latency profiling, and behavioral observations:

- [`notebooks/01_grounding_dino_sandbox.ipynb`](notebooks/01_grounding_dino_sandbox.ipynb) — GroundingDINO behavior, threshold sweeps, compound query analysis
- [`notebooks/02_sam2_sandbox.ipynb`](notebooks/02_sam2_sandbox.ipynb) — SAM 2 mask quality, multi-mask selection, latency scaling with bbox count
- [`notebooks/03_sam2_video_sandbox.ipynb`](notebooks/03_sam2_video_sandbox.ipynb) — SAM 2 Video Predictor, memory bank propagation, multi-object tracking, latency findings vs image predictor

---

## Testing

The project includes 98 unit and integration tests covering type validation, the component registry, tracker behavior, and pipeline orchestration logic. Real model inference is excluded from automated tests (uses mock detectors and segmenters) to keep the suite fast and CI-friendly.

```bash
pip install -e ".[dev]"
pytest tests/
# 98 passed in ~3s
```

---

## Roadmap

- [x] Image pipeline (detector + segmenter)
- [x] CLI entry points (`ovp-image`, `ovp-video`, `ovp-benchmark`)
- [x] Pydantic-based type system with validation
- [x] Strategy pattern + registry for component swapping
- [x] **VideoPipeline** with keyframe strategy
- [x] **ByteTracker integration** for persistent track IDs
- [x] **Track-aware visualization** (track IDs in annotated output)
- [x] **Quantitative benchmarks** on COCO val2017 (mAP, mIoU, FPS)
- [x] **Test coverage** with pytest (98 tests, mock-based)
- [ ] **fp16 inference** with proper autocast (currently fp32 baseline)
- [ ] **NMS post-processing** for crowded scenes
- [ ] **CLIP-based attribute filter** as optional pipeline stage
- [ ] **SAM 2 video predictor integration** for inter-keyframe mask propagation
- [ ] **Docker container** for reproducible deployment
- [ ] **CI/CD** with GitHub Actions

---

## Tech Stack

- **Models:** GroundingDINO (IDEA-Research), SAM 2 (Meta AI)
- **Tracking:** ByteTrack (via supervision)
- **ML framework:** PyTorch 2.11 + CUDA 13.0
- **HF integration:** transformers 4.57
- **Visualization:** supervision (Roboflow)
- **Type system:** Pydantic v2
- **CLI:** Typer + Rich
- **Config:** OmegaConf
- **Video I/O:** OpenCV
- **Benchmarking:** pycocotools + torchmetrics

---

## License

MIT

---

## Author

**Emin Çağan Apaydın** — Computer Engineering, Istanbul Okan University  
GitHub: [@e-cagan](https://github.com/e-cagan)