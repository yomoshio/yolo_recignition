import os
import cv2
import shutil
import albumentations as A


def check_and_copy_annotations(frame_dir, annotation_dir, output_annotation_dir):
    os.makedirs(output_annotation_dir, exist_ok=True)
    missing_annotations = []
    
    for frame in os.listdir(frame_dir):
        if frame.endswith(".jpg"):
            annotation_path = os.path.join(annotation_dir, frame.replace(".jpg", ".txt"))
            output_path = os.path.join(output_annotation_dir, frame.replace(".jpg", ".txt"))
            
            if os.path.exists(annotation_path):
                shutil.copyfile(annotation_path, output_path)
            else:
                missing_annotations.append(frame)
    
    if missing_annotations:
        print(f"Предупреждение: Для {len(missing_annotations)} кадров отсутствуют аннотации: {missing_annotations[:5]}...")
    else:
        print(f"Все аннотации скопированы в {output_annotation_dir}")


def augment_data(image_dir, annotation_dir, output_image_dir, output_annotation_dir):
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_annotation_dir, exist_ok=True)
    
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=30, p=0.3),
        A.RandomCrop(height=512, width=512, p=0.3),
        A.Resize(height=640, width=640)
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
    
    for image_name in os.listdir(image_dir):
        if image_name.endswith(".jpg"):
            image_path = os.path.join(image_dir, image_name)
            annotation_path = os.path.join(annotation_dir, image_name.replace(".jpg", ".txt"))
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Ошибка: Не удалось загрузить {image_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            bboxes = []
            class_labels = []
            if os.path.exists(annotation_path):
                with open(annotation_path, "r") as f:
                    for line in f:
                        try:
                            class_id, x_center, y_center, width, height = map(float, line.strip().split())
                            bboxes.append([x_center, y_center, width, height])
                            class_labels.append(int(class_id))
                        except ValueError:
                            print(f"Ошибка в аннотации: {annotation_path}")
                            continue
            
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_labels = augmented["class_labels"]
            
            aug_image_name = f"aug_{image_name}"
            aug_image_path = os.path.join(output_image_dir, aug_image_name)
            cv2.imwrite(aug_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            
            aug_annotation_path = os.path.join(output_annotation_dir, aug_image_name.replace(".jpg", ".txt"))
            with open(aug_annotation_path, "w") as f:
                for bbox, label in zip(aug_bboxes, aug_labels):
                    x_center, y_center, width, height = bbox
                    f.write(f"{label} {x_center} {y_center} {width} {height}\n")
    
    print(f"Аугментированные данные сохранены в {output_image_dir} и {output_annotation_dir}")