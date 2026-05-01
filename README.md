# Open-Vocabulary Perception (OVP)

> A modular pipeline for **text-prompted object detection and segmentation**, combining GroundingDINO with SAM 2 for class-agnostic mask generation. Built with a strict no-vibe-coding philosophy: every component is hand-written, every design decision documented.

![Hero](outputs/demo_01_hero_cats.jpg)

*Single command: detect cats and remote controls, return pixel-precise masks.*

```bash
ovp-image -i input.jpg -p "cat,remote control" -o output.jpg
```

---

## Why This Project

Open-vocabulary perception is the natural evolution of closed-class detection. Instead of training on a fixed taxonomy, the model accepts arbitrary text queries at inference time. This project demonstrates how to compose two state-of-the-art foundation models — **GroundingDINO** (text → bounding box) and **SAM 2** (bounding box → pixel mask) — into a unified, production-ready pipeline.

Built and tested on RTX 3050 Ti Laptop (4GB VRAM). Both models fit comfortably (~526MB total VRAM when loaded together).

---

## Architecture

```
text prompts ──► GroundingDinoDetector ──► Detection (bbox + label + score)
                                                 │
                                                 ▼
RGB image    ───► Sam2Segmenter ◄─────────  BoundingBox
                       │
                       ▼
                  Mask (binary pixel array)
                       │
                       ▼
              ImagePipeline orchestrator
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
├── pipeline/         # ImagePipeline (VideoPipeline planned)
├── viz/              # FrameAnnotator
└── scripts/          # CLI entry points
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

### Quick Start

Detect and segment using the CLI:

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

### Programmatic API

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
print(f"Latency: {result.inference_times}")  # {'detector': X, 'segmenter': Y}

annotated = FrameAnnotator().annotate(image, result)
Image.fromarray(annotated).save("output.jpg")
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

## Limitations

This project is honest about what works and what doesn't.

### Compound queries are weakly enforced

GroundingDINO treats attributes as soft hints, not hard constraints. Querying for `"person wearing white shirt"` will likely match all persons regardless of clothing.

![Compound](outputs/demo_04_compound_queries.jpg)

For attribute-sensitive use cases, consider noun-only prompts followed by a downstream classifier (e.g., CLIP-based filtering on detection crops).

### Crowded scenes lack post-processing

The pipeline currently lacks proper NMS or duplicate suppression. In dense scenes, multiple overlapping bounding boxes for the same person can produce visually noisy mask overlays.

![Crowded](outputs/demo_03_crowded_scene.jpg)

A future version will integrate class-aware NMS in the pipeline orchestrator.

### Latency is not real-time

Single-image inference is ~600ms on RTX 3050 Ti (fp32). This is below real-time video thresholds; the planned `VideoPipeline` will use a keyframe strategy (run detector every N frames) combined with SAM 2's video memory bank or a tracker for in-between frames.

---

## Performance

Benchmarks on RTX 3050 Ti Laptop GPU (4GB VRAM), fp32, single 640×480 image:

| Component | Latency (warm) | VRAM |
|---|---|---|
| GroundingDINO-tiny (detector) | ~430 ms | 337 MB |
| SAM 2.1 Hiera-tiny (segmenter, 4 boxes) | ~160 ms | 121 MB |
| **End-to-end pipeline** | **~600 ms** | **~526 MB** |

SAM 2's promptable architecture decouples image encoding (~140 ms, runs once) from mask decoding (~4 ms per box, scales with prompt count). Dense scenes with 10–20 detections do not significantly slow segmentation.

---

## Sandbox Notebooks

Both models were systematically tested before being wrapped in production classes. The notebooks document threshold sensitivity, latency profiling, and behavioral observations:

- [`notebooks/01_grounding_dino_sandbox.ipynb`](notebooks/01_grounding_dino_sandbox.ipynb) — GroundingDINO behavior, threshold sweeps, compound query analysis
- [`notebooks/02_sam2_sandbox.ipynb`](notebooks/02_sam2_sandbox.ipynb) — SAM 2 mask quality, multi-mask selection, latency scaling with bbox count

---

## Roadmap

- [x] Image pipeline (detector + segmenter)
- [x] CLI entry point
- [x] Pydantic-based type system with validation
- [x] Strategy pattern + registry for component swapping
- [ ] **VideoPipeline** with keyframe strategy
- [ ] **ByteTracker integration** for persistent track IDs
- [ ] **NMS post-processing** for crowded scenes
- [ ] **fp16 inference** with proper autocast (currently fp32 baseline)
- [ ] **CLIP-based attribute filter** as optional pipeline stage

---

## Tech Stack

- **Models:** GroundingDINO (IDEA-Research), SAM 2 (Meta AI)
- **ML framework:** PyTorch 2.11 + CUDA 13.0
- **HF integration:** transformers 4.57
- **Visualization:** supervision (Roboflow)
- **Type system:** Pydantic v2
- **CLI:** Typer + Rich
- **Config:** OmegaConf

---

## License

MIT

---

## Author

**Emin Çağan Apaydın** — Computer Engineering, Istanbul Okan University  
GitHub: [@e-cagan](https://github.com/e-cagan)