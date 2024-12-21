import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Загрузка данных о винах
wine_data = datasets.load_wine()
features = wine_data.data  # Характеристики вин
labels = wine_data.target  # Типы вин

# Разделение данных на обучающую и тестовую выборки
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Создание и обучение модели классификации
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)  # Многослойный перцептрон
model.fit(train_features, train_labels)  # Обучение модели

# Прогнозирование на тестовых данных
predicted_labels = model.predict(test_features)

# Вычисление точности
precision = accuracy_score(test_labels, predicted_labels)

# Вывод результата
print(f'Точность классификации: {precision:.2f}')
