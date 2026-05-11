import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Параметры по умолчанию
IMG_SIZE = (128, 128)      # размер изображения
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = "model.h5"
TRAIN_SPLIT = 0.8          # доля данных для обучения (остальное валидация)

def prepare_data(data_dir):
    """
    Загрузка данных из каталогов с помощью ImageDataGenerator.
    Ожидается структура: data_dir/class1/, data_dir/class2/, ...
    Возвращает train_generator, val_generator, class_names.
    """
    # Генератор с аугментацией для тренировочных данных
    train_datagen = ImageDataGenerator(
        rescale=1./255,               # нормализация
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=TRAIN_SPLIT  # разделение на train/val
    )

    # Для валидации только нормализация
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=TRAIN_SPLIT)

    # Генератор для обучения (подкаталог train)
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # Генератор для валидации
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    class_names = list(train_generator.class_indices.keys())
    print("Обнаружены классы:", class_names)

    return train_generator, val_generator, class_names

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(data_dir, epochs=EPOCHS, model_path=MODEL_PATH):
    """
    Обучение модели на данных из data_dir.
    Сохраняет лучшую модель в model_path.
    """
    print("Загрузка данных...")
    train_gen, val_gen, class_names = prepare_data(data_dir)

    print("Создание модели...")
    input_shape = (*IMG_SIZE, 3)  # цветные изображения (RGB)
    model = build_model(input_shape, len(class_names))
    model.summary()

    # Коллбэки: остановка при отсутствии улучшений и сохранение лучшей модели
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, save_best_only=True, verbose=1)
    ]

    print("Начало обучения...")

    # Получить метки всех тренировочных изображений
    train_labels = train_gen.classes  # целочисленные метки
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = dict(enumerate(class_weights))

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Графики обучения
    plot_training_history(history)

    # Оценка на валидационных данных
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"\nИтоговая точность на валидации: {val_acc:.4f}")

    # Сохраним имена классов для последующего использования
    with open('class_names.txt', 'w') as f:
        for name in class_names:
            f.write(name + '\n')

    print(f"Модель сохранена в {model_path}")
    return model, class_names

def plot_training_history(history):
    """Отображение графиков потерь и точности."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['loss'], label='Train loss')
    ax1.plot(history.history['val_loss'], label='Val loss')
    ax1.set_title('Loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='Train acc')
    ax2.plot(history.history['val_accuracy'], label='Val acc')
    ax2.set_title('Accuracy')
    ax2.legend()

    plt.show()

def predict_image(image_path, model_path=MODEL_PATH, class_names_path='class_names.txt'):
    """
    Классификация одного изображения.
    Возвращает предсказанный класс и вероятности.
    """
    # Загружаем модель
    model = load_model(model_path)

    # Загружаем имена классов
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    # Загрузка и предобработка изображения
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0   # нормализация

    # Предсказание
    predictions = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Визуализация
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f})")
    plt.axis('off')
    plt.show()

    # Вывод вероятностей по классам
    print("Вероятности:")
    for name, prob in zip(class_names, predictions):
        print(f"  {name}: {prob:.4f}")

    return predicted_class, predictions

def main():
    parser = argparse.ArgumentParser(description="Классификатор изображений (MLP)")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Подкоманда для обучения
    train_parser = subparsers.add_parser('train', help='Обучить модель')
    train_parser.add_argument('--data_dir', required=True, help='Путь к корневому каталогу с подкаталогами классов')
    train_parser.add_argument('--epochs', type=int, default=EPOCHS, help='Количество эпох')
    train_parser.add_argument('--model_path', default=MODEL_PATH, help='Путь для сохранения модели')

    # Подкоманда для предсказания
    predict_parser = subparsers.add_parser('predict', help='Распознать изображение')
    predict_parser.add_argument('--image_path', required=True, help='Путь к изображению')
    predict_parser.add_argument('--model_path', default=MODEL_PATH, help='Путь к сохранённой модели')
    predict_parser.add_argument('--class_names', default='class_names.txt', help='Файл с именами классов')

    args = parser.parse_args()

    if args.command == 'train':
        train_model(args.data_dir, args.epochs, args.model_path)
    elif args.command == 'predict':
        predict_image(args.image_path, args.model_path, args.class_names)

if __name__ == '__main__':
    main()