from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Загрузка датасета wine
wine = datasets.load_wine()

# Подготовка данных
X = wine.data
y = wine.target

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)

# Прогнозирование и оценка точности
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Вывод точности на экран
print(f'Accuracy: {accuracy:.2f}')