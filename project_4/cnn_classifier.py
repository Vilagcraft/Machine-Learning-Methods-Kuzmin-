import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
import shutil

# ======================== НАСТРОЙКИ ========================
DATA_DIR = './cifar10_4classes'          # корневая папка с данными
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
IMG_SIZE = (64, 64)                      # увеличим размер для ярче выраженного преимущества
BATCH_SIZE = 32
EPOCHS_CNN = 15
EPOCHS_MLP = 30                          # MLP нужно больше эпох
NUM_CLASSES = 4
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat']   # 4 класса из CIFAR-10
CIFAR10_CLASSES = [0, 1, 2, 3]           # индексы выбранных классов

# ======================== ПОДГОТОВКА ДАННЫХ ========================
def prepare_data():
    """Загружает CIFAR-10, выбирает 4 класса и сохраняет изображения в структуру папок."""
    if os.path.exists(DATA_DIR):
        print(f"Папка {DATA_DIR} уже существует. Для пересоздания удалите её вручную.")
        return

    # Загрузка CIFAR-10
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0).flatten()

    # Выбираем только нужные классы
    mask = np.isin(y, CIFAR10_CLASSES)
    x = x[mask]
    y = y[mask]

    # Преобразуем индексы классов в 0..NUM_CLASSES-1
    for new_id, old_id in enumerate(CIFAR10_CLASSES):
        y[y == old_id] = new_id

    # Разделение на train/val (80% / 20%)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

    # Создание папок
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, class_name), exist_ok=True)

    # Функция сохранения изображений
    def save_images(images, labels, base_dir):
        for i, (img, label) in enumerate(zip(images, labels)):
            class_dir = os.path.join(base_dir, CLASS_NAMES[label])
            # Изменяем размер до IMG_SIZE для единообразия и лучшего обучения
            img_resized = tf.image.resize(img, IMG_SIZE).numpy().astype(np.uint8)
            img_pil = Image.fromarray(img_resized)
            img_pil.save(os.path.join(class_dir, f'{i}.jpg'))

    print("Сохранение обучающей выборки...")
    save_images(x_train, y_train, TRAIN_DIR)
    print("Сохранение валидационной выборки...")
    save_images(x_val, y_val, VAL_DIR)
    print(f"Данные подготовлены в {DATA_DIR}")

# ======================== СОЗДАНИЕ МОДЕЛЕЙ ========================
def create_cnn_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=NUM_CLASSES):
    """Свёрточная нейронная сеть."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_mlp_model(input_shape=(IMG_SIZE[0] * IMG_SIZE[1] * 3,), num_classes=NUM_CLASSES):
    """Многослойный персептрон (полносвязная сеть)."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ======================== ОБУЧЕНИЕ И ОЦЕНКА ========================
def train_model(model, model_name, train_gen, val_gen, epochs, save_best=True):
    """Обучает модель и возвращает историю, сохраняет лучшую."""
    checkpoint = keras.callbacks.ModelCheckpoint(
        f'best_{model_name}.h5', save_best_only=True, monitor='val_accuracy', mode='max'
    )
    early_stop = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    if save_best:
        model.save(f'{model_name}_final.h5')
        print(f"Модель {model_name} сохранена.")
    return history

def plot_comparison(hist_cnn, hist_mlp):
    """Сравнительные графики точности и потерь."""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist_cnn.history['accuracy'], label='CNN train')
    plt.plot(hist_cnn.history['val_accuracy'], label='CNN val')
    plt.plot(hist_mlp.history['accuracy'], label='MLP train')
    plt.plot(hist_mlp.history['val_accuracy'], label='MLP val')
    plt.title('Точность')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist_cnn.history['loss'], label='CNN train')
    plt.plot(hist_cnn.history['val_loss'], label='CNN val')
    plt.plot(hist_mlp.history['loss'], label='MLP train')
    plt.plot(hist_mlp.history['val_loss'], label='MLP val')
    plt.title('Потери')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ======================== ПРЕДСКАЗАНИЕ НОВЫХ ИЗОБРАЖЕНИЙ ========================
def predict_image(model, image_path, class_names):
    """Загружает и классифицирует одно изображение."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array, verbose=0)
    class_idx = np.argmax(pred[0])
    confidence = np.max(pred[0])
    print(f"Результат: {class_names[class_idx]} (уверенность: {confidence:.2f})")
    return class_idx, confidence

# ======================== ОСНОВНОЙ БЛОК ========================
if __name__ == '__main__':
    # 1. Подготовка данных (если ещё нет)
    if not os.path.exists(DATA_DIR):
        print("Данные не найдены. Выполняем подготовку...")
        prepare_data()
    else:
        print("Данные уже существуют. Пропускаем подготовку.")

    # 2. Генераторы данных с аугментацией
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        classes=CLASS_NAMES
    )
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        classes=CLASS_NAMES,
        shuffle=False
    )

    # 3. Обучение CNN
    print("\n=== Обучение свёрточной сети ===")
    cnn_model = create_cnn_model()
    history_cnn = train_model(cnn_model, "cnn", train_generator, val_generator, EPOCHS_CNN)

    # 4. Обучение MLP для сравнения
    print("\n=== Обучение полносвязной сети (MLP) ===")
    # Преобразуем генераторы в плоские векторы для MLP
    def flatten_generator(gen, target_size):
        steps = len(gen)
        x, y = [], []
        for i in range(steps):
            batch_x, batch_y = gen[i]
            x.append(batch_x.reshape(batch_x.shape[0], -1))
            y.append(batch_y)
        return np.concatenate(x, axis=0), np.concatenate(y, axis=0)

    x_train_flat, y_train_flat = flatten_generator(train_generator, IMG_SIZE)
    x_val_flat, y_val_flat = flatten_generator(val_generator, IMG_SIZE)

    mlp_model = create_mlp_model(input_shape=(x_train_flat.shape[1],))
    history_mlp = mlp_model.fit(
        x_train_flat, y_train_flat,
        validation_data=(x_val_flat, y_val_flat),
        epochs=EPOCHS_MLP,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )
    mlp_model.save('mlp_final.h5')
    print("MLP модель сохранена.")

    # 5. Сравнение результатов
    print("\n=== Сравнение точности на валидации ===")
    cnn_val_acc = max(history_cnn.history['val_accuracy'])
    mlp_val_acc = max(history_mlp.history['val_accuracy'])
    print(f"CNN лучшая валидационная точность: {cnn_val_acc:.4f}")
    print(f"MLP лучшая валидационная точность: {mlp_val_acc:.4f}")
    print(f"Преимущество CNN: {(cnn_val_acc - mlp_val_acc) * 100:.2f}%")

    plot_comparison(history_cnn, history_mlp)

    # 6. Распознавание нового изображения (пример)
    print("\n=== Распознавание нового изображения ===")
    # Укажите путь к своему изображению или используйте тестовое из валидационной выборки
    sample_image_path = os.path.join(VAL_DIR, CLASS_NAMES[0], os.listdir(os.path.join(VAL_DIR, CLASS_NAMES[0]))[0])
    print(f"Загружаем тестовое изображение: {sample_image_path}")
    predict_image(cnn_model, sample_image_path, CLASS_NAMES)