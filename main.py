from pathlib import Path
from utils.annotate import augment_data, check_and_copy_annotations
from utils.extract import extract_frames
from utils.hyperparams import optimize_hyperparameters
from utils.baseline import train_yolo
from utils.dataset import create_dataset_structure
from utils.graph import plot_yolo_metrics
from utils.video import create_inference_video

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    video_paths = [
        base_dir / "videos/video1.mov",
        base_dir / "videos/video2.mov",
        base_dir / "videos/video3.mov",
        base_dir / "videos/video4.mov",
        base_dir / "videos/video5.mov",
        base_dir / "videos/video6.mov",
    ]
    frame_dir = base_dir / "frames"
    annotation_dir = base_dir / "annotations"
    aug_image_dir = base_dir / "annotations/augmented/images"
    aug_annotation_dir = base_dir / "annotations/augmented/labels"
    dataset_dir = base_dir / "annotations/dataset"
    experiment_dir = base_dir / "annotations/experiments"

    while True:
        print("\n=== Меню пайплайна обработки данных и обучения ===")
        print("1. Извлечение кадров из видео")
        print("2. Проверка и копирование аннотаций")
        print("3. Аугментация данных")
        print("4. Создание структуры датасета")
        print("5. Обучение базовой модели")
        print("6. Оптимизация гиперпараметров (1-я итерация) на пустой модели")
        print("7. Оптимизация гиперпараметров (2-я итерация) на пустой модели")
        print("8. Оптимизация гиперпараметров (1-я итерация) на базовой модели")
        print("9. Оптимизация гиперпараметров (2-я итерация) на базовой модели")
        print("10. Построение графиков метрик")
        print("11. Создание видео с инференсом")
        print("0. Выход")
        
        choice = input("Выберите действие (0-11): ")
        
        if choice == "1":
            extract_frames(video_paths, frame_dir)
            print("Кадры извлечены. Перейдите к аннотации в LabelImg и выберите следующий шаг.")
        elif choice == "2":
            check_and_copy_annotations(frame_dir, annotation_dir, dataset_dir / "train/labels")
        elif choice == "3":
            augment_data(frame_dir, annotation_dir, aug_image_dir, aug_annotation_dir)
        elif choice == "4":
            create_dataset_structure(aug_image_dir, aug_annotation_dir, dataset_dir)
        elif choice == "5":
            results_baseline = train_yolo(dataset_dir / "data.yaml", experiment_dir / "baseline")
        elif choice == "6":
            results_opt1 = optimize_hyperparameters(dataset_dir / "data.yaml", experiment_dir / "optimized", experiment_dir, iteration=1)
        elif choice == "7":
            results_opt2 = optimize_hyperparameters(dataset_dir / "data.yaml", experiment_dir / "optimized", experiment_dir, iteration=2)
        elif choice == "8":
            results_opt1 = optimize_hyperparameters(dataset_dir / "data.yaml", experiment_dir / "optimized", experiment_dir, iteration=3)
        elif choice == "9":
            results_opt2 = optimize_hyperparameters(dataset_dir / "data.yaml", experiment_dir / "optimized", experiment_dir, iteration=4)
        elif choice == "10":
            plot_yolo_metrics(experiment_dir, experiment_dir / "baseline_metrics.png")
            plot_yolo_metrics(experiment_dir, experiment_dir / "opt1_metrics.png")
            plot_yolo_metrics(experiment_dir, experiment_dir / "opt2_metrics.png") 
            plot_yolo_metrics(experiment_dir, experiment_dir / "opt3_metrics.png")
            plot_yolo_metrics(experiment_dir, experiment_dir / "opt4_metrics.png") 
        elif choice == "11":
            create_inference_video(
                experiment_dir / "baseline9/weights/best.pt",
                video_paths, 
                base_dir / "inference.mp4"
            )
        elif choice == "0":
            print("Выход из программы.")
            break
        else:
            print("Неверный выбор. Введите число от 0 до 11.")