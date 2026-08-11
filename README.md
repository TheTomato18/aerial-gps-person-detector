# aerial-gps-person-detector

A two-stage pipeline for search-and-rescue style drone work:

1. **Detect** — a YOLO26n object detector trained to spot people in aerial/drone imagery.
2. **Geolocate** — run that detector over a drone video, and use the drone's flight log (position, altitude, compass heading) to convert each detection's pixel location into a real GPS coordinate, plotted on an interactive map.

## Project structure

| Path | Purpose |
| --- | --- |
| [train_model.py](train_model.py) | Trains the YOLO26n detector on the HERIDAL dataset |
| [drone_geolocation.py](drone_geolocation.py) | Runs a trained detector over a drone video + flight log and renders a GPS map of detections |
| `requirements.txt` | Python dependencies for both stages |
| `yolo26n.pt` | Pretrained COCO checkpoint used as the training starting point (git-ignored — downloaded automatically on first run) |
| `heridal_yolo26_dir/` | Dataset directory (git-ignored — see [Dataset](#dataset)) |
| `runs/` | Training outputs: weights, curves, metrics (git-ignored) |
| `detections/` | Snapshot image of each confirmed detection (git-ignored) |
| `flight_map.html` | Generated map, the output of the geolocation stage (git-ignored) |
| [NOTICE](NOTICE) | Attribution for the upstream work this project ports (see [Acknowledgments](#acknowledgments)) |

Only the two scripts and their supporting files are tracked; weights, datasets, footage, and generated outputs are all git-ignored, since they're large and may contain real GPS data.

## Dataset

Trained on [HERIDAL](https://universe.roboflow.com/nageeb-moin/heridal-lrbkc-c7e3z) (1,600 images, single `person` class), exported from Roboflow in YOLO format. Images are auto-oriented and resized to 640x640; no augmentation is applied at export time. License: CC BY 4.0.

The dataset is not included in this repo (it's git-ignored). Download it from the link above in YOLO format and unpack it into `heridal_yolo26_dir/`, split into `train`, `valid`, and `test` sets. The export includes a `data.yaml`; [train_model.py](train_model.py) expects it at `heridal_yolo26_dir/data.yaml`.

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

It uses Ultralytics' defaults for everything the script doesn't set explicitly (640px images, device auto-selected, no early stopping until 100 epochs of no improvement, no RAM caching); batch size is auto-selected to fit available GPU memory. Adjust the `model.train(...)` call in [train_model.py](train_model.py) directly if you want to tune these — e.g. pin a GPU with `device=0`, raise resolution with `imgsz=1280`, or set a tighter `patience` for early stopping.

Training artifacts (weights, curves, confusion matrix, per-batch samples) are written to `runs/detect/` and are git-ignored. The trained weights end up at `runs/detect/heridal_yolo26n/weights/best.pt`.

## Geolocation

[drone_geolocation.py](drone_geolocation.py) takes a drone video and its flight-log CSV, runs a trained detector frame-by-frame, and converts each detection's pixel location into a GPS coordinate using the drone's recorded position, altitude, and compass heading. Detections that land close together across multiple frames are merged into a single marker, which is only plotted once it's been confirmed by enough frames. The result is written out as a standalone interactive HTML map (flight path + detection markers), alongside a saved snapshot image of each confirmed detection.

```bash
python drone_geolocation.py path/to/video.mp4 path/to/flightlog.csv --weights runs/detect/heridal_yolo26n/weights/best.pt -o flight_map.html
```

- `--weights` — path to a trained model's `.pt` file (produced by `train_model.py`). If omitted, only the flight path is plotted, with no detections.
- `--conf` (default `0.4`) — detection confidence threshold.
- `--interval` (default `0.2`) — seconds of video between detection passes.
- `-o` / `--output` (default `flight_map.html`) — output HTML map path.
- `--snapshots` (default `detections`) — directory the detection snapshots are written into.
- `--no-snapshots` — skip writing snapshot files (thumbnails are still embedded in the map).

The flight-log CSV must have `isVideo`, `latitude`, `longitude`, `ascent(feet)`, and `compass_heading(degrees)` columns sampled at 10 rows/sec, matching typical DJI flight-log (Airdata) exports. Only the first continuous block of rows flagged `isVideo` is used, so the log is aligned with the video that was recorded during it.

The pixel-to-GPS projection assumes a ~59° diagonal camera field of view (measured for a DJI Mavic Air); adjust the `fov_atan` calculation in `process_video()` if you're flying a different drone/camera.

Clustering behaviour — how close two detections must be to merge (20 m), and how many frames must agree before a marker is plotted (3) — is set in the `CONFIG` dict at the top of [drone_geolocation.py](drone_geolocation.py).

### Detection snapshots

Every confirmed detection is saved as `detections/detection_NN.jpg`: a crop of the frame where the detector was most confident about that cluster, with the bounding box drawn on and enough surrounding terrain to judge the hit in context (4x the box size, at least 256px). The same image is embedded as a thumbnail in that marker's map popup — as base64, so the HTML stays a single self-contained file — along with the coordinates, detection count, source frame/timestamp, and confidence. The numbering in the filenames matches the popups, so a marker on the map can be traced back to its image on disk.

Crop sizing and thumbnail width are also set in that `CONFIG` dict.

## Results

Latest run (`heridal_yolo26n`, 100 epochs) on the validation split:

| Metric | Value |
| --- | --- |
| Precision | 0.84 |
| Recall | 0.61 |
| mAP50 | 0.73 |
| mAP50-95 | 0.30 |

`train_model.py` also evaluates the held-out `test` split and prints its mAP50 / mAP50-95 at the end of the run; those are the numbers to quote when comparing against other models, since the validation split is seen during training.

## Requirements

- Python 3 with the packages in `requirements.txt` installed
- A CUDA-capable GPU is strongly recommended for training; requires a CUDA-enabled PyTorch build (e.g. for Blackwell GPUs like the RTX 5070, `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`)

## Acknowledgments

The geolocation stage is a Python port of [`src/renderMap.js`](https://github.com/roboflow/dji-aerial-georeferencing/blob/main/src/renderMap.js) from Roboflow's [dji-aerial-georeferencing](https://github.com/roboflow/dji-aerial-georeferencing), used under the Apache License 2.0. The approach for turning a detection's pixel position into a GPS coordinate is theirs: derive the ground distance a frame covers from the camera's field of view and the drone's altitude, project the detection's offset from frame center into a bearing and distance, then follow a rhumb line out from the drone. The clustering thresholds in `CONFIG` — 20 m merge distance, 3 frames to confirm — come from that project too, values included. Their write-up, [*Georeferencing Drone Videos*](https://blog.roboflow.com/georeferencing-drone-videos/), explains the method well.

[drone_geolocation.py](drone_geolocation.py) is modified from the original: rewritten in Python, with Turf.js swapped for local rhumb-line and haversine implementations, Mapbox GL for Folium, Roboflow's hosted inference API for a local Ultralytics checkpoint behind a pluggable `DetectionModel`, flight-log parsing rebuilt on pandas, and per-detection snapshots added. [NOTICE](NOTICE) records this in full.

Also with thanks to the authors of the HERIDAL dataset (see [Dataset](#dataset)), and to [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO.

## License

This project is MIT licensed — see [LICENSE](LICENSE).

[drone_geolocation.py](drone_geolocation.py) derives from Apache-2.0 licensed work, so that license's terms apply to the parts taken from it; a copy is included as [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0), and the attribution and record of changes is in [NOTICE](NOTICE).

The HERIDAL dataset is separately licensed CC BY 4.0 and is not distributed here.
