from ultralytics import YOLO


def main():
    model = YOLO("yolo26n.pt")

    model.train(
        data="./heridal_yolo26_dir/data.yaml",
        epochs=100,
        imgsz=1280,
        batch=16,
        device=0,
        workers=4,
        patience=20,
        cache="ram",
        name="heridal_yolo26n",
    )

    # routine check - val split
    val_metrics = model.val()  # dataset/settings remembered from training
    print("val mAP50-95:", val_metrics.box.map)
    print("val mAP50:", val_metrics.box.map50)
    print("val mAP75:", val_metrics.box.map75)
    print("val per-class mAP50-95:", val_metrics.box.maps)
    print("val per-image metrics:", val_metrics.box.image_metrics)

    # final, one-time - untouched test split, this is the number to report
    test_metrics = model.val(data="./heridal_yolo26_dir/data.yaml", split="test")
    print("test mAP50-95:", test_metrics.box.map)
    print("test mAP50:", test_metrics.box.map50)

    # inference over test images, saved to disk automatically (unique filenames)
    results = model(
        "./heridal_yolo26_dir/test/images",
        stream=True,
        save=True,
    )

    for i, result in enumerate(results):
        if i < 5:              # only pop up windows for the first 5
            result.show()
        boxes = result.boxes


if __name__ == "__main__":
    main()