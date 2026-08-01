# aerial-gps-person-detector

A two-stage pipeline for search-and-rescue style drone work:

1. **Detect** — a YOLO26n object detector trained to spot people in aerial/drone imagery.
2. **Geolocate** — run that detector over a drone video, and use the drone's flight log (position, altitude, compass heading) to convert each detection's pixel location into a real GPS coordinate, plotted on an interactive map.

## Project structure

| Path | Purpose |
| --- | --- |
| [train_model.py](train_model.py) | Trains the YOLO26n detector on the HERIDAL dataset |
| [drone_geolocation.py](drone_geolocation.py) | Runs a trained detector over a drone video + flight log and renders a GPS map of detections |
| `yolo26n.pt` | Pretrained COCO checkpoint used as the training starting point |
| `heridal_yolo26_dir/` | Dataset directory (git-ignored — see [Dataset](#dataset)) |
| `runs/` | Training outputs: weights, curves, metrics (git-ignored) |
| `requirements.txt` | Python dependencies for both stages |

## Dataset

Trained on [HERIDAL](https://universe.roboflow.com/nageeb-moin/heridal-lrbkc-c7e3z) (1,600 images, single `person` class), exported from Roboflow in YOLO format. Images are auto-oriented and resized to 640x640; no augmentation is applied at export time. License: CC BY 4.0.

The dataset is not included in this repo (it's git-ignored). Download it from the link above in YOLO format and unpack it into `heridal_yolo26_dir/`, split into `train`, `valid`, and `test` sets (see `data.yaml`).

## Setup

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended for training (see [Requirements](#requirements)).

## Training

[train_model.py](train_model.py) fine-tunes `yolo26n.pt` on the dataset for 100 epochs, then reports validation and held-out test metrics and runs inference over the test images:

```bash
python train_model.py
```

It uses Ultralytics' defaults for everything not explicitly set above (auto batch sizing, 640px images, device auto-selected, no early stopping until 100 epochs of no improvement, no RAM caching). Adjust the `model.train(...)` call in `train_model.py` directly if you want to tune these — e.g. pin a GPU with `device=0`, raise resolution with `imgsz=1280`, or set a tighter `patience` for early stopping.

Training artifacts (weights, curves, confusion matrix, per-batch samples) are written to `runs/detect/` and are git-ignored. The trained weights end up at `runs/detect/heridal_yolo26n/weights/best.pt`.

## Geolocation

[drone_geolocation.py](drone_geolocation.py) takes a drone video and its flight-log CSV, runs a trained detector frame-by-frame, and converts each detection's pixel location into a GPS coordinate using the drone's recorded position, altitude, and compass heading. Detections that land close together across multiple frames are merged into a single marker, which is only plotted once it's been confirmed by enough frames. The result is written out as a standalone interactive HTML map (flight path + detection markers).

```bash
python drone_geolocation.py path/to/video.mp4 path/to/flightlog.csv --weights runs/detect/heridal_yolo26n/weights/best.pt -o flight_map.html
```

- `--weights` — path to a trained model's `.pt` file (produced by `train_model.py`). If omitted, only the flight path is plotted, with no detections.
- `--conf` (default `0.4`) — detection confidence threshold.
- `--interval` (default `0.2`) — seconds of video between detection passes.
- `-o` / `--output` (default `flight_map.html`) — output HTML map path.

The flight-log CSV must have `isVideo`, `latitude`, `longitude`, `ascent(feet)`, and `compass_heading(degrees)` columns sampled at 10 rows/sec, matching typical DJI flight-log (Airdata) exports.

The pixel-to-GPS projection assumes a ~59° diagonal camera field of view (measured for a DJI Mavic Air); adjust the `fov_atan` calculation in `process_video()` if you're flying a different drone/camera.

## Results

Latest run (`heridal_yolo26n`, 100 epochs) on the validation split:

| Metric | Value |
| --- | --- |
| Precision | 0.84 |
| Recall | 0.61 |
| mAP50 | 0.73 |
| mAP50-95 | 0.30 |

## Requirements

- Python 3 with the packages in `requirements.txt` installed
- A CUDA-capable GPU is strongly recommended for training; requires a CUDA-enabled PyTorch build (e.g. for Blackwell GPUs like the RTX 5070, `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`)

## License

MIT - see [LICENSE](LICENSE). Note the HERIDAL dataset itself is separately licensed CC BY 4.0.
