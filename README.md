An AI-based system using computer vision and deep learning to detect industrial defects, assess severity, and assign quality grades. It improves inspection accuracy, reduces human error, saves time, and enables consistent quality control in manufacturing.
----------------
⚠️ Unable to view the file directly?
If the file doesn't display properly on GitHub, you can still download it easily.

How to Download?

1)Open the file on GitHub.
2)Click View raw.
3)The raw file will open in your browser.
4)Right-click and select Save As to download it.
5)You can then open the downloaded file locally on your computer.
----------------
# DEFECTAI

### Explainable AI-Based Framework for Automated Industrial Defect Detection, Severity Assessment and Product Quality Grading

DefectAI is a complete, runnable industrial visual-inspection platform. It detects
surface defects on products from **images, videos, or a live webcam**, scores how
severe each defect is with a fully transparent formula, assigns a **quality grade
(A/B/C/REJECT)**, and explains *why* it made that decision — all through a premium,
single-page web dashboard.

It ships with a **synthetic demonstration dataset and a pre-trainable model**, so it
runs immediately after installation. You do not need to find or download any dataset.

> **Honesty note (please read):** this is an academic / demonstration prototype.
> The dataset is procedurally generated (not real industrial photography), and the
> model is a CPU-friendly classical-computer-vision + machine-learning ensemble, not
> a deep neural network. Both are clearly labeled as such throughout the app. See
> [Limitations](#limitations) below.

---

## Features

- **Six defect classes**: Scratch, Dent, Hole, Crack, Corrosion, Surface Discoloration — plus Normal/No-Defect
- **Three inspection modes**: image upload, video upload, and live webcam
- **Bounding boxes + confidence** for every detected defect
- **Transparent severity scoring (0–100)** with a documented, configurable formula
- **Automatic quality grading**: Grade A / B / C / REJECT with PASS / REVIEW / REJECT decisions
- **Explainable AI**: anomaly heatmap, annotated detection image, and a fact-grounded textual explanation for every decision
- **Inspection history** stored in SQLite, with search / filter / sort / delete
- **PDF inspection reports**, generated on demand
- **Dashboard & analytics** with live charts (defects by type, severity distribution, pass/reject trend, quality grades)
- **Dataset & Model pages** — regenerate the demo dataset or retrain the model from the UI
- **Runtime-configurable thresholds** (confidence, severity, grading) from the Settings page
- **CPU-only**: no GPU required, works on a normal laptop
- **One-command startup**: `python run.py`

---

## Architecture

```
Industrial Product
      │
      ▼
Image / Video / Camera Capture
      │
      ▼
Preprocessing            (resize, denoise, CLAHE — ai/preprocessing.py)
      │
      ▼
AI Detection              (region proposal + classifier — ai/detector.py)
      │
      ▼
Defect Classification     (7 classes: normal + 6 defect types)
      │
      ▼
Severity Assessment       (transparent weighted formula — ai/severity.py)
      │
      ▼
Quality Grading           (A / B / C / REJECT — ai/grading.py)
      │
      ▼
Explainability             (heatmap + annotated image + text — ai/explainability.py)
      │
      ▼
Dashboard / Final Decision (Flask API + web UI — backend/, frontend)
```

### Why classical CV + RandomForest instead of YOLO / PyTorch?

The project brief's preferred stack was YOLO + PyTorch. This environment had **no
GPU, no internet access, and no PyTorch/Ultralytics installation available**, so a
CPU-only, dependency-light architecture was used instead:

1. **Region proposal** (`ai/detector.py::propose_regions`) — classical background
   subtraction (median-blur background estimate) + saturation-anomaly detection
   finds candidate defect regions without any learned model.
2. **Classification** (`ai/training.py`, `ai/features.py`) — each candidate region is
   converted into a 13-dimensional handcrafted feature vector (area, texture, edge
   density, color, circularity, etc.) and classified by a **RandomForest ensemble**
   trained on the synthetic dataset.

The `Detector` class's public interface (`detect(image) -> list[dict]`) is stable and
isolated from the rest of the app, so a real Ultralytics-YOLO backend can be dropped
in later as a drop-in replacement — see [Future Enhancements](#future-enhancements).

**Measured performance** (on a held-out split of the synthetic dataset, computed
honestly — not fabricated): **~84% end-to-end detection hit-rate**, **~99% specificity**
on defect-free samples, and classifier metrics of **~91% accuracy / ~84% macro-F1**
on the region-classification sub-task. Run `python -m ai.training` to reproduce.

---

## Installation

```bash
pip install -r requirements.txt
```

(If your system requires it: `pip install -r requirements.txt --break-system-packages`)

Requires Python 3.9+. No GPU, no CUDA, no external dataset download needed.

## Run

```bash
python run.py
```

This single command will:

1. Check that all Python dependencies are installed.
2. Create the required directories (`data/`, `models/`, `uploads/`, `outputs/`, `reports/`, `database/`).
3. Generate the **220-sample synthetic demonstration dataset** if it doesn't exist yet.
4. Train the DefectNet model if a trained model isn't saved yet.
5. Start the web application and open your browser to `http://127.0.0.1:5050`.

First run takes about 30–60 seconds (dataset generation + training). Every run after
that starts in a couple of seconds, since the dataset and model are cached on disk.

---

## Demo

- **Dashboard → "Try Demo Inspection"** — instantly runs a full inspection on a
  random defective sample from the dataset, no upload needed.
- **Image tab** — drag & drop a JPG/PNG/WEBP, or use one of your own product photos.
  You can also preview/download sample images from the **Dataset** page.
- **Video tab** — upload an MP4/AVI/MOV/MKV/WEBM clip; frames are sampled and
  processed (capped for CPU responsiveness), and an annotated output video is
  generated.
- **Live Camera tab** — click **Start Camera**, allow webcam access, and watch live
  bounding boxes + severity/grade overlay. Click **Capture** to save a full
  inspection (with heatmap + report) to history.

---

## Dataset

DefectAI uses a **Synthetic Demonstration Dataset** (`dataset/generator.py`):
procedurally generated industrial-surface images across 10 product categories
(metal panel, automotive part, pipe, machine component, electronic housing, bottle,
sheet metal, bearing, gear, industrial component), with realistic-looking
procedurally rendered defects (irregular scratches, elliptical dents with
shading, dark holes with inner shadow, branching cracks, rust-textured corrosion
blobs, and soft color-patch discoloration).

- 220 total samples, 70/15/15 train/val/test split
- ~28% of samples are defect-free ("normal")
- Every defect has a ground-truth bounding box, label, and area percentage
- Regenerate anytime from the **Dataset** page or `python -m dataset.generator`

**Limitation**: this is not real industrial photography. It is useful for
demonstrating the complete pipeline end-to-end, not for validating real-world
detection accuracy.

## AI Model

- **Name**: DefectNet v1.0 (Prototype / Demonstration Model)
- **Architecture**: classical-CV region proposal + RandomForest classifier (see
  [Architecture](#architecture) above)
- **Classes (7)**: normal, scratch, dent, hole, crack, corrosion, discoloration
- Retrain anytime from the **Model** page or `python -m ai.training`
- All thresholds (confidence, IoU, severity, grading) live in `config.yaml` and are
  also editable at runtime from the **Settings** page

## Explainable AI

Because the architecture has no convolutional feature maps, classic Grad-CAM does
not apply. Instead, DefectAI provides an honest, equivalent explanation for *this*
architecture:

- **Heatmap** — a false-color visualization of the same background-subtraction /
  saturation-anomaly signal that drove region proposal (i.e., literally what made
  the algorithm look twice at that region).
- **Detection overlay** — bounding boxes, defect labels, confidence, and a
  severity-coded border color.
- **Textual explanation** — a deterministic, template-filled summary built directly
  from the detection/severity/grading output (never free-form generated text), so
  it can never state a fact the pipeline didn't actually compute.

Severity itself is fully transparent — see the formula below — and every quality
decision states its reason (e.g. *"Severity (87/100) exceeds the rejection
threshold (75)"*).

```
Severity Score = Area Impact × 0.35
               + Defect Type Risk × 0.25
               + Location Risk × 0.20
               + Defect Count × 0.10
               + Model Confidence × 0.10        (normalised to 0–100)
```

---

## Project Structure

```
industrial-defect-ai/
├── run.py                  # one-command launcher
├── requirements.txt
├── config.yaml              # all tunable thresholds
├── backend/
│   ├── app.py               # Flask API + routing
│   ├── database.py          # SQLite inspection history
│   ├── reports.py           # PDF report generation
│   ├── config_loader.py
│   └── static/               # frontend (HTML/CSS/vanilla JS, no build step)
├── ai/
│   ├── preprocessing.py
│   ├── detector.py           # region proposal + classification
│   ├── features.py           # handcrafted CV feature extraction
│   ├── training.py           # trains + evaluates the RandomForest model
│   ├── severity.py
│   ├── grading.py
│   └── explainability.py
├── dataset/
│   └── generator.py          # synthetic dataset + defect simulators
├── data/                     # generated dataset (train/val/test + metadata)
├── models/                   # trained model + metrics.json
├── uploads/ / outputs/ / reports/ / database/
└── tests/                    # unit + API tests
```

## API

```
GET    /api/health
GET    /api/model
POST   /api/model/train
GET    /api/dataset
GET    /api/dataset/preview
POST   /api/dataset/generate
POST   /api/inspect/image
POST   /api/inspect/video
POST   /api/inspect/frame          (live camera; capture:true persists it)
POST   /api/inspect/demo
GET    /api/inspections
GET    /api/inspections/<id>
DELETE /api/inspections/<id>
GET    /api/dashboard
GET    /api/report/<id>
GET    /api/settings
POST   /api/settings
```

## Testing

```bash
python -m unittest discover -s tests -v
```

42 tests cover preprocessing, the detector (region proposal + classification),
severity scoring, quality grading, the dataset generator, the SQLite database
layer, and the full API (including file-validation and error-handling paths) via
Flask's test client.

---

## Limitations

- The dataset is **synthetic**, not real industrial photography. Real deployment
  requires a genuinely representative, domain-specific dataset with real defect
  photography, validated against real quality-control outcomes.
- The detector is a classical-CV + RandomForest ensemble, chosen because this
  environment had no GPU/PyTorch/internet access — not a deep neural network. Its
  ~84% end-to-end hit-rate on the synthetic test set is a demonstration figure, not
  a production accuracy claim.
- Region proposal and severity's "location risk" are heuristic, not learned from
  real failure-mode data.
- Video and live-camera inference use frame sampling / frame skipping for CPU
  responsiveness, not full framerate processing.
- No authentication/authorization — this is a local demo tool, not hardened for
  multi-tenant or production line deployment.

## Future Enhancements

- Swap in a trained Ultralytics YOLO (or other CNN) detector behind the same
  `Detector.detect()` interface once GPU/training infrastructure is available
- Larger, real, labeled industrial defect datasets per product line
- Additional defect categories (delamination, warping, foreign-object debris, etc.)
- Direct integration with production-line cameras / PLC systems
- Edge deployment (Jetson / industrial PC) with quantized models
- IoT integration for multi-station factory-floor monitoring
- Cloud-based fleet monitoring and centralized analytics across sites
- Grad-CAM / attention-based explainability once a CNN backbone is in place
- Proper authentication, audit logging, and role-based access for production use

---

## License / Attribution

Demonstration / academic prototype. Dataset generator, model, backend, and frontend
were built specifically for this project.
