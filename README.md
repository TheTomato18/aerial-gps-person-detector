# aerial-gps-person-detector

A YOLO26 object detector trained to spot people in aerial/drone imagery, aimed at search-and-rescue style use cases.

## Dataset

Trained on [HERIDAL](https://universe.roboflow.com/nageeb-moin/heridal-lrbkc-c7e3z) (1,600 images, single `person` class), exported from Roboflow in YOLO format. Images are auto-oriented and resized to 640x640; no augmentation is applied at export time. License: CC BY 4.0.

The dataset is not included in this repo (it's git-ignored). Download it from the link above in YOLO format and unpack it into `heridal_yolo26_dir/`, split into `train`, `valid`, and `test` sets (see `data.yaml`).

## Training

[main.py](main.py) trains a YOLO26n model on the dataset, then reports validation and held-out test metrics and runs inference over the test images:

```bash
python main.py
```

Key settings: 100 epochs, 1280px images, batch size 16, RAM caching, early stopping patience of 20. Adjust these directly in `main.py` for your hardware (the default assumes a single CUDA GPU at `device=0`).

Training artifacts (weights, curves, confusion matrix, per-batch samples) are written to `runs/detect/` and are git-ignored.

## Results

Latest run (`heridal_yolo26n`, 100 epochs) on the validation split:

| Metric | Value |
| --- | --- |
| Precision | 0.84 |
| Recall | 0.61 |
| mAP50 | 0.73 |
| mAP50-95 | 0.30 |

## Requirements

- Python with [ultralytics](https://pypi.org/project/ultralytics/) installed
- A CUDA-capable GPU is strongly recommended for training

## License

MIT - see [LICENSE](LICENSE). Note the HERIDAL dataset itself is separately licensed CC BY 4.0.
