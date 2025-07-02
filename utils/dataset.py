import os
import yaml
import shutil
from sklearn.model_selection import train_test_split


def create_dataset_structure(image_dir, annotation_dir, dataset_dir):
    images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    train_images, temp_images = train_test_split(images, test_size=0.3, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)
    
    splits = {"train": train_images, "val": val_images, "test": test_images}
    for split in splits:
        split_image_dir = os.path.join(dataset_dir, split, "images")
        split_label_dir = os.path.join(dataset_dir, split, "labels")
        os.makedirs(split_image_dir, exist_ok=True)
        os.makedirs(split_label_dir, exist_ok=True)
        
        for image in splits[split]:
            src_image = os.path.join(image_dir, image)
            src_label = os.path.join(annotation_dir, image.replace(".jpg", ".txt"))
            dst_image = os.path.join(split_image_dir, image)
            dst_label = os.path.join(split_label_dir, image.replace(".jpg", ".txt"))
            
            shutil.copyfile(src_image, dst_image)
            if os.path.exists(src_label):
                shutil.copyfile(src_label, dst_label)
    
    data_config = {
        "train": os.path.join(dataset_dir, "train/images"),
        "val": os.path.join(dataset_dir, "val/images"),
        "test": os.path.join(dataset_dir, "test/images"),
        "nc": 11,
        "names": ["steak", "salad", "soup", "cake", "tea", "empty_plate_steak", "empty_plate_salad", "empty_plate_soup", "empty_plate_cake", "cup", "empty_cup"]
    }
    with open(os.path.join(dataset_dir, "data.yaml"), "w") as f:
        yaml.dump(data_config, f)
    
    print(f"Датасет создан в {dataset_dir}")
