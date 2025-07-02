import os
import shutil
from pathlib import Path

# Настройки
base_dir = Path("D:/dish_recignition/annotations")  
base_file = "video6_frame_01809.txt" 
start_frame = 1809  
end_frame = 2157    
interval = 3      


base_path = base_dir / base_file


if not base_path.exists():
    print(f"Ошибка: Базовый файл {base_path} не найден!")
    exit()


with open(base_path, "r") as f:
    base_content = f.readlines()


for frame_num in range(start_frame, end_frame + 1, interval):
    formatted_frame = f"{frame_num:05d}"
    target_file = base_dir / f"video6_frame_{formatted_frame}.txt"
    
    with open(target_file, "w") as f:
        f.writelines(base_content)
    
    print(f"Создан файл: {target_file}")

print("Копирование завершено!")