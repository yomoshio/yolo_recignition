from ultralytics import YOLO


def train_yolo(dataset_yaml, experiment_name, epochs=100, batch_size=4, img_size=640):
    model = YOLO("../yolo11s.pt")
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        name=experiment_name,
        device=0,  
        workers=4,
        patience=15,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,
        freeze=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=30.0,
        translate=0.2,
        scale=0.9,
        shear=0.2,
        perspective=0.001,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        auto_augment="randaugment",
        multi_scale=True,
        val=True
    )
    return results
