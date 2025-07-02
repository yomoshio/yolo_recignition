import os
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split


def extract_frames(video_paths, output_dir, frame_interval=3):
    os.makedirs(output_dir, exist_ok=True)
    total_frames = 0
    
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Ошибка: Не удалось открыть {video_path}")
            continue
        count = 0
        video_name = Path(video_path).stem
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"{video_name}_frame_{count:05d}.jpg")
                cv2.imwrite(frame_path, frame)
                total_frames += 1
            count += 1
        cap.release()
    
    print(f"Извлечено {total_frames} кадров в {output_dir}")
    return total_frames
