import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import os
import glob

class KMeansApp:
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.X_scaled = None          # стандартизированные признаки (обучающие)
        self.X_pca = None             # проекция на 2 компоненты (для визуализации)
        self.pca = None               # объект PCA (сохраняется для новых точек)
        self.feature_names = None
        self.df = None

    def list_available_datasets(self):
        """Возвращает список Excel/CSV файлов в текущей папке."""
        files = glob.glob("*.xlsx") + glob.glob("*.csv")
        return files

    def choose_dataset(self):
        """Интерактивный выбор: использовать существующий файл или сгенерировать новый."""
        print("\n=== Выбор источника данных ===")
        print("1. Использовать существующий датасет (Excel/CSV)")
        print("2. Сгенерировать новый синтетический датасет")
        choice = input("Ваш выбор (1 или 2): ").strip()

        if choice == "1":
            files = self.list_available_datasets()
            if not files:
                print("В папке не найдено файлов .xlsx или .csv. Будет сгенерирован синтетический датасет.")
                return self.generate_synthetic_data()
            print("\nНайденные файлы:")
            for i, f in enumerate(files, 1):
                print(f"  {i}. {f}")
            while True:
                try:
                    idx = int(input(f"Выберите номер файла (1-{len(files)}): ")) - 1
                    if 0 <= idx < len(files):
                        filename = files[idx]
                        break
                    else:
                        print("Неверный номер.")
                except ValueError:
                    print("Введите число.")
            return self.load_custom_dataset(filename)
        else:
            return self.generate_synthetic_data()

    def load_custom_dataset(self, filename):
        """Загружает пользовательский датасет из Excel или CSV."""
        if filename.endswith('.xlsx'):
            df = pd.read_excel(filename)
        else:
            df = pd.read_csv(filename)

        # Оставляем только числовые столбцы
        df = df.select_dtypes(include=[np.number])
        if df.shape[1] < 5:
            raise ValueError(f"Датасет содержит только {df.shape[1]} числовых признаков. Нужно минимум 5.")
        if df.shape[0] < 50:
            raise ValueError(f"Датасет содержит {df.shape[0]} объектов. Нужно минимум 50.")
        
        print(f"Загружен датасет '{filename}': {df.shape[0]} объектов, {df.shape[1]} признаков.")
        self.feature_names = df.columns.tolist()
        self.df = df
        return df

    def generate_synthetic_data(self, n_samples=100, n_features=5):
        """Генерирует синтетические данные и сохраняет в CSV."""
        print(f"Генерируем синтетические данные: {n_samples} объектов, {n_features} признаков.")
        np.random.seed(self.random_state)
        data = pd.DataFrame({
            f'Признак_{i+1}': np.random.normal(0, 1, n_samples) for i in range(n_features)
        })
        # Добавляем структуру, чтобы кластеры были различимы
        data['Признак_1'] += 2 * (np.random.randint(0, 3, n_samples) - 1)
        data['Признак_2'] += 1.5 * (np.random.randint(0, 3, n_samples) - 1)
        filename = "synthetic_data.csv"
        data.to_csv(filename, index=False)
        print(f"Синтетические данные сохранены в '{filename}'.")
        self.feature_names = data.columns.tolist()
        self.df = data
        return data

    def preprocess(self, df):
        """Стандартизация признаков."""
        X = df.values
        self.X_scaled = self.scaler.fit_transform(X)
        joblib.dump(self.scaler, 'scaler.pkl')
        print("Признаки стандартизированы (среднее=0, дисперсия=1).")

    def train(self):
        """Обучает KMeans и вычисляет PCA для визуализации."""
        if self.X_scaled is None:
            raise RuntimeError("Сначала вызовите preprocess().")
        
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        self.model.fit(self.X_scaled)
        
        # Оценка качества
        inertia = self.model.inertia_
        sil_score = silhouette_score(self.X_scaled, self.model.labels_)
        print(f"\nОбучение завершено. Кластеров: {self.n_clusters}")
        print(f"Инерция: {inertia:.2f}")
        print(f"Коэффициент силуэта: {sil_score:.3f}")
        
        joblib.dump(self.model, 'kmeans_model.pkl')
        
        # Распределение объектов по кластерам
        labels = self.model.labels_
        unique, counts = np.unique(labels, return_counts=True)
        print("Распределение объектов по кластерам:")
        for k, cnt in zip(unique, counts):
            print(f"  Кластер {k}: {cnt} объектов")
        
        # Построение PCA для визуализации (если признаков > 2)
        if self.X_scaled.shape[1] > 2:
            self.pca = PCA(n_components=2)
            self.X_pca = self.pca.fit_transform(self.X_scaled)
            joblib.dump(self.pca, 'pca.pkl')
        else:
            # Если признаков 1 или 2, используем их напрямую
            self.X_pca = self.X_scaled
            self.pca = None
            print("Признаков <= 2, PCA не применяется.")

    def visualize_clusters(self):
        """Визуализирует кластеры (и центроиды) в 2D."""
        if self.X_pca is None:
            print("Нет данных для визуализации. Сначала обучите модель.")
            return
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(self.X_pca[:, 0], self.X_pca[:, 1], 
                              c=self.model.labels_, cmap='viridis', alpha=0.7)
        
        # Отображаем центроиды в том же PCA-пространстве
        if self.pca is not None:
            centroids_pca = self.pca.transform(self.model.cluster_centers_)
        else:
            centroids_pca = self.model.cluster_centers_
        
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                    marker='X', s=200, c='red', label='Центроиды')
        plt.title("Визуализация кластеров (PCA)")
        plt.xlabel("Компонента 1")
        plt.ylabel("Компонента 2")
        plt.colorbar(scatter, label='Кластер')
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict_new_object(self, new_values, visualize=True):
        """
        Определяет кластер для нового объекта.
        Если visualize=True, показывает точку на графике вместе с обучающими данными.
        """
        if self.model is None:
            raise RuntimeError("Модель не обучена.")
        
        # Загружаем стандартизатор
        scaler = joblib.load('scaler.pkl')
        new_array = np.array(new_values).reshape(1, -1)
        new_scaled = scaler.transform(new_array)
        cluster = self.model.predict(new_scaled)[0]
        distances = self.model.transform(new_scaled)[0]
        
        print(f"\nРезультат предсказания для объекта {new_values}:")
        print(f"  Принадлежит кластеру: {cluster}")
        print("  Расстояния до центроидов кластеров:")
        for i, d in enumerate(distances):
            print(f"    Кластер {i}: {d:.4f}")
        
        if visualize and self.X_pca is not None:
            self._visualize_new_point(new_scaled, cluster, distances)
        
        return cluster

    def _visualize_new_point(self, new_scaled, cluster, distances):
        """Отображает новый объект на существующем PCA-графике."""
        # Преобразуем новую точку в PCA-пространство
        if self.pca is not None:
            new_pca = self.pca.transform(new_scaled)
        else:
            new_pca = new_scaled
        
        plt.figure(figsize=(8, 6))
        # Обучающие точки
        scatter = plt.scatter(self.X_pca[:, 0], self.X_pca[:, 1], 
                              c=self.model.labels_, cmap='viridis', alpha=0.6)
        # Центроиды
        if self.pca is not None:
            centroids_pca = self.pca.transform(self.model.cluster_centers_)
        else:
            centroids_pca = self.model.cluster_centers_
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                    marker='X', s=200, c='red', label='Центроиды')
        # Новая точка
        plt.scatter(new_pca[0, 0], new_pca[0, 1], 
                    marker='*', s=300, c='lime', edgecolors='black', 
                    label=f'Новый объект (кластер {cluster})')
        
        plt.title("Визуализация с новым объектом")
        plt.xlabel("Компонента 1")
        plt.ylabel("Компонента 2")
        plt.colorbar(scatter, label='Кластер')
        plt.legend()
        plt.grid(True)
        plt.show()

    def interactive_predict(self):
        """Интерактивный ввод нового объекта."""
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
                self.predict_new_object(values, visualize=True)
            except ValueError:
                print("Ошибка: введите числа, разделённые запятыми.")

    def run(self):
        """Основной цикл приложения."""
        print("="*50)
        print("KMeans кластеризация с выбором датасета и визуализацией")
        print("="*50)
        
        # 1. Выбор датасета
        df = self.choose_dataset()
        
        # 2. Предобработка
        self.preprocess(df)
        
        # 3. Обучение
        self.train()
        
        # 4. Визуализация обучающих данных
        viz = input("\nПоказать 2D-визуализацию кластеров? (y/n): ").lower()
        if viz == 'y':
            self.visualize_clusters()
        
        # 5. Интерактивное предсказание с визуализацией новых объектов
        self.interactive_predict()

if __name__ == "__main__":
    app = KMeansApp(n_clusters=3)   # можно изменить число кластеров
    app.run()