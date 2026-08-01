"""
Trains a YOLO26n person detector on the HERIDAL dataset.

Fine-tunes `yolo26n.pt` against `heridal_yolo26_dir/data.yaml` (see
README.md's Dataset section), reports validation and test metrics, then
runs inference over the test images as a sanity check.

Dependencies: pip install -r requirements.txt
"""

from ultralytics import YOLO


def main():
    model = YOLO("yolo26n.pt")

    model.train(
        data="./heridal_yolo26_dir/data.yaml",
        epochs=100,
        batch=-1,  # auto-select batch size to fit available GPU memory
        name="heridal_yolo26n",  # fixed path; see drone_geolocation.py --weights
    )

    # val split (weights/data remembered from training)
    val_metrics = model.val()
    print("val mAP50-95:", val_metrics.box.map)
    print("val mAP50:", val_metrics.box.map50)
    print("val mAP75:", val_metrics.box.map75)
    print("val per-class mAP50-95:", val_metrics.box.maps)
    print("val per-image metrics:", val_metrics.box.image_metrics)

    # held-out test split, evaluated once — this is the number to report
    test_metrics = model.val(data="./heridal_yolo26_dir/data.yaml", split="test")
    print("test mAP50-95:", test_metrics.box.map)
    print("test mAP50:", test_metrics.box.map50)

    # visual sanity check over the test images
    results = model(
        "./heridal_yolo26_dir/test/images",
        stream=True,
        save=True,
    )

    for i, result in enumerate(results):
        if i < 5:  # preview the first 5
            result.show()


if __name__ == "__main__":
    main()