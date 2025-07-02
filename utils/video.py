import cv2
from ultralytics import YOLO
from pathlib import Path


def create_inference_video(model_path, video_paths, output_path):
    model = YOLO(model_path)
    
    temp_videos = []
    
    width, height, fps = None, None, None
    
    for idx, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Ошибка: Не удалось открыть {video_path}")
            continue
        
        if width is None or height is None or fps is None:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        temp_output = f"temp_inference_{idx}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, imgsz=640, conf=0.5)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
        
        cap.release()
        out.release()
        temp_videos.append(temp_output)
        print(f"Обработано видео: {video_path}, сохранено во временный файл: {temp_output}")
    

    if temp_videos:
        final_out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for temp_video in temp_videos:
            cap = cv2.VideoCapture(temp_video)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                final_out.write(frame)
            cap.release()
        
        final_out.release()
        print(f"Итоговое видео сохранено в {output_path}")
        
        for temp_video in temp_videos:
            Path(temp_video).unlink()
            print(f"Удален временный файл: {temp_video}")
    else:
        print("Ошибка: Не удалось обработать ни одно видео.")