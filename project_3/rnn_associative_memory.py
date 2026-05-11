import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
import os
from PIL import Image

# ------------------------------------------------------------
# 1. Генерация 10 различных эталонных изображений 10x10
# ------------------------------------------------------------
def generate_ideal_images():
    images = []
    # 1. Крест (обе диагонали)
    img1 = np.zeros((10,10))
    for i in range(10):
        img1[i,i] = 1
        img1[i,9-i] = 1
    images.append(img1)
    
    # 2. Рамка
    img2 = np.zeros((10,10))
    img2[0,:] = img2[-1,:] = img2[:,0] = img2[:,-1] = 1
    images.append(img2)
    
    # 3. Вертикальная полоса
    img3 = np.zeros((10,10))
    img3[:,4:6] = 1
    images.append(img3)
    
    # 4. Горизонтальная полоса
    img4 = np.zeros((10,10))
    img4[4:6,:] = 1
    images.append(img4)
    
    # 5. Шахматная доска
    img5 = np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            if (i+j)%2==0: img5[i,j]=1
    images.append(img5)
    
    # 6. Диагональ (главная)
    img6 = np.zeros((10,10))
    for i in range(10): img6[i,i]=1
    images.append(img6)
    
    # 7. Круг (приближённо)
    img7 = np.zeros((10,10))
    for i in range(10):
        for j in range(10):
            if (i-4.5)**2 + (j-4.5)**2 < 12:
                img7[i,j]=1
    images.append(img7)
    
    # 8. Буква 'T'
    img8 = np.zeros((10,10))
    img8[0,:] = 1          # верхняя горизонталь
    img8[:,4] = 1          # вертикаль по центру
    images.append(img8)
    
    # 9. Уголок (левый верхний)
    img9 = np.zeros((10,10))
    img9[0:5,0:5] = 1
    images.append(img9)
    
    # 10. Зигзаг
    img10 = np.zeros((10,10))
    for i in range(10):
        img10[i, i%2 + 4] = 1
    images.append(img10)
    
    return np.array(images)

# ------------------------------------------------------------
# 2. Добавление шума (инвертирование с вероятностью p)
# ------------------------------------------------------------
def add_noise(img, p=0.2):
    noisy = img.copy()
    mask = np.random.random(img.shape) < p
    noisy[mask] = 1 - noisy[mask]
    return noisy

# ------------------------------------------------------------
# 3. Формирование обучающей выборки
# ------------------------------------------------------------
def build_dataset(ideal_imgs, samples_per_img=250, noise_level=0.25):
    X, Y = [], []
    for img in ideal_imgs:
        for _ in range(samples_per_img):
            noisy = add_noise(img, noise_level)
            X.append(noisy.flatten())
            Y.append(img.flatten())
    return np.array(X), np.array(Y)

# ------------------------------------------------------------
# 4. Создание упрощённой модели (один слой RNN + Dense)
# ------------------------------------------------------------
def build_simple_model():
    model = Sequential()
    # Вход: последовательность из 100 временных шагов, 1 признак
    model.add(SimpleRNN(64, input_shape=(100, 1)))
    model.add(Dense(100, activation='sigmoid'))  # выход 100 пикселей
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ------------------------------------------------------------
# 5. Восстановление изображения из массива или файла
# ------------------------------------------------------------
def restore_from_array(model, noisy_flat):
    # noisy_flat: массив из 100 значений (0/1)
    inp = noisy_flat.reshape(1, 100, 1)
    pred = model.predict(inp, verbose=0)[0]  # (100,)
    restored = (pred > 0.5).astype(np.float32).reshape(10,10)
    return restored

def load_image_as_binary(filepath, target_size=(10,10), thresh=128):
    try:
        img = Image.open(filepath).convert('L')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        return (arr > thresh).astype(np.float32)
    except Exception as e:
        print("Ошибка загрузки:", e)
        return None

# ------------------------------------------------------------
# 6. Отображение трёх картинок
# ------------------------------------------------------------
def show_triplet(orig, noisy, restored, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(9,3))
    axes[0].imshow(orig, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Оригинал")
    axes[1].imshow(noisy, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Зашумлённое")
    axes[2].imshow(restored, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("Восстановленное")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 7. Основная программа
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Генерация 10 эталонных изображений...")
    ideal_imgs = generate_ideal_images()
    print(f"Создано {ideal_imgs.shape[0]} образов 10x10.")
    
    print("Создание обучающей выборки (250 зашумлённых копий каждого, шум 25%)...")
    X, Y = build_dataset(ideal_imgs, samples_per_img=250, noise_level=0.25)
    X = X.reshape(-1, 100, 1)
    Y = Y.reshape(-1, 100)
    print(f"Размер выборки: {X.shape[0]} примеров")
    
    print("Создание упрощённой модели RNN...")
    model = build_simple_model()
    model.summary()
    
    print("Обучение модели (20 эпох)...")
    history = model.fit(X, Y, epochs=40, batch_size=64, validation_split=0.1, verbose=1)
    
    # Демонстрация на первом эталоне
    print("\nТест на первом эталонном образе (крест):")
    test_orig = ideal_imgs[0]
    test_noisy = add_noise(test_orig, p=0.3)
    test_restored = restore_from_array(model, test_noisy.flatten())
    show_triplet(test_orig, test_noisy, test_restored, "Тест на кресте")
    
    # Запрос пользовательского файла
    ans = input("\nХотите восстановить своё изображение? (y/n): ").strip().lower()
    if ans == 'y':
        path = input("Введите полный путь к файлу (например, C:/my_pic.png): ").strip()
        user_img = load_image_as_binary(path)
        if user_img is not None:
            # Добавляем шум к пользовательскому изображению (чтобы показать восстановление)
            noisy_user = add_noise(user_img, p=0.3)
            restored_user = restore_from_array(model, noisy_user.flatten())
            show_triplet(user_img, noisy_user, restored_user, "Ваше изображение")
        else:
            print("Не удалось загрузить файл. Проверьте путь и расширение.")
    
    print("Программа завершена.")