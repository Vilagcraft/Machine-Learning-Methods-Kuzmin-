import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import os

class KMeansApp:
    """
    Приложение для кластеризации данных методом KMeans.
    """
    def __init__(self, data_path='data.csv', n_clusters=3, random_state=42):
        self.data_path = data_path
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.X_scaled = None
        self.feature_names = None

    def generate_sample_data(self, n_samples=100, n_features=5):
        """Генерирует синтетические данные и сохраняет в CSV, если файл не существует."""
        if os.path.exists(self.data_path):
            print(f"Загружаем существующие данные из {self.data_path}")
            return pd.read_csv(self.data_path)
        
        print(f"Генерируем синтетические данные: {n_samples} объектов, {n_features} признаков.")
        np.random.seed(self.random_state)
        # Создаём осмысленные признаки (пример: финансовые метрики)
        data = pd.DataFrame({
            'Признак_1': np.random.normal(100, 15, n_samples),
            'Признак_2': np.random.normal(50, 8, n_samples),
            'Признак_3': np.random.normal(200, 30, n_samples),
            'Признак_4': np.random.normal(10, 2, n_samples),
            'Признак_5': np.random.normal(5, 1, n_samples),
        })
        # Добавляем небольшую корреляцию, чтобы кластеры были различимы
        data['Признак_1'] += 0.5 * data['Признак_2']
        data['Признак_3'] -= 0.3 * data['Признак_4']
        data.to_csv(self.data_path, index=False)
        print(f"Данные сохранены в {self.data_path}")
        return data

    def load_data(self):
        """Загружает данные из CSV."""
        df = pd.read_csv(self.data_path)
        self.feature_names = df.columns.tolist()
        print(f"Загружено {df.shape[0]} объектов, {df.shape[1]} признаков.")
        if df.shape[1] < 5:
            raise ValueError("Данные должны содержать минимум 5 признаков (столбцов).")
        if df.shape[0] < 50:
            raise ValueError("Данные должны содержать минимум 50 объектов (строк).")
        return df

    def preprocess(self, df):
        """Стандартизация признаков."""
        X = df.values
        self.X_scaled = self.scaler.fit_transform(X)
        # Сохраняем стандартизатор для новых данных
        joblib.dump(self.scaler, 'scaler.pkl')
        print("Признаки стандартизированы (среднее=0, дисперсия=1).")

    def train(self):
        """Обучает модель KMeans и оценивает качество."""
        if self.X_scaled is None:
            raise RuntimeError("Данные не предобработаны. Вызовите preprocess() сначала.")
        
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        self.model.fit(self.X_scaled)
        
        # Оценка качества
        inertia = self.model.inertia_
        sil_score = silhouette_score(self.X_scaled, self.model.labels_)
        print(f"\nОбучение завершено. Кластеров: {self.n_clusters}")
        print(f"Инерция (внутрикластерная сумма квадратов): {inertia:.2f}")
        print(f"Коэффициент силуэта: {sil_score:.3f} (чем ближе к 1, тем лучше)")
        
        # Сохраняем модель
        joblib.dump(self.model, 'kmeans_model.pkl')
        print("Модель сохранена в 'kmeans_model.pkl'")
        
        # Вывод распределения объектов по кластерам
        labels = self.model.labels_
        unique, counts = np.unique(labels, return_counts=True)
        print("Распределение объектов по кластерам:")
        for k, cnt in zip(unique, counts):
            print(f"  Кластер {k}: {cnt} объектов")

    def visualize_2d(self):
        """Визуализация кластеров в 2D с помощью PCA (если признаков >2)."""
        if self.X_scaled is None or self.model is None:
            print("Сначала обучите модель.")
            return
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.model.labels_, cmap='viridis', alpha=0.7)
        plt.scatter(self.model.cluster_centers_[:, 0], self.model.cluster_centers_[:, 1],
                    marker='X', s=200, c='red', label='Центроиды')
        plt.title("Визуализация кластеров (PCA)")
        plt.xlabel("Первая главная компонента")
        plt.ylabel("Вторая главная компонента")
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict_new_object(self, new_values):
        """
        Определяет кластер для нового объекта.
        new_values: список или массив значений признаков в том же порядке.
        """
        if self.model is None:
            raise RuntimeError("Модель не обучена. Вызовите train() сначала.")
        
        # Загружаем стандартизатор
        scaler = joblib.load('scaler.pkl')
        new_array = np.array(new_values).reshape(1, -1)
        new_scaled = scaler.transform(new_array)
        cluster = self.model.predict(new_scaled)[0]
        distances = self.model.transform(new_scaled)[0]  # расстояния до центроидов
        
        print(f"\nРезультат предсказания для объекта {new_values}:")
        print(f"  Принадлежит кластеру: {cluster}")
        print("  Расстояния до центроидов кластеров:")
        for i, d in enumerate(distances):
            print(f"    Кластер {i}: {d:.4f}")
        return cluster

    def interactive_predict(self):
        """Интерактивный ввод нового объекта с клавиатуры."""
        print("\n=== Режим предсказания нового объекта ===")
        print(f"Введите {len(self.feature_names)} значений признаков через запятую.")
        print(f"Признаки: {', '.join(self.feature_names)}")
        while True:
            user_input = input("\nВведите значения (или 'q' для выхода): ")
            if user_input.lower() == 'q':
                break
            try:
                values = [float(x.strip()) for x in user_input.split(',')]
                if len(values) != len(self.feature_names):
                    print(f"Ошибка: нужно {len(self.feature_names)} значений, получено {len(values)}.")
                    continue
                self.predict_new_object(values)
            except ValueError:
                print("Ошибка: введите числа, разделённые запятыми.")

    def run(self):
        """Основной цикл приложения."""
        print("="*50)
        print("KMeans кластеризация")
        print("="*50)
        
        # 1. Данные
        df = self.generate_sample_data()
        df = self.load_data()
        
        # 2. Предобработка
        self.preprocess(df)
        
        # 3. Обучение (число кластеров можно задать при создании объекта)
        self.train()
        
        # 4. Визуализация (по желанию)
        viz = input("\nПоказать 2D-визуализацию кластеров? (y/n): ").lower()
        if viz == 'y':
            self.visualize_2d()
        
        # 5. Интерактивное предсказание
        self.interactive_predict()

if __name__ == "__main__":
    # Параметры: можно изменить n_clusters, путь к данным
    app = KMeansApp(data_path='sample_data.csv', n_clusters=3)
    app.run()