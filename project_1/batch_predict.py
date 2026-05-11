import os
import subprocess

new_image_dir = r"D:\Мой Диск\University\Методы машинного обучения (Кузьмин)\project1\new_image"
model_path = "model.h5"
class_names_file = "class_names.txt"

for filename in os.listdir(new_image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(new_image_dir, filename)
        print(f"\n=== Анализ {filename} ===")
        subprocess.run([
            "python", "app.py", "predict",
            "--image_path", img_path,
            "--model_path", model_path,
            "--class_names", class_names_file
        ])
