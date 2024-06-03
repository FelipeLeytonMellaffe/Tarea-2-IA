import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Cargar el dataset de Nursery desde UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data"
column_names = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "class"]
data = pd.read_csv(url, names=column_names)

# Convertir las características categóricas a valores numéricos
label_encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Separar características (X) y etiquetas (y)
X = data.drop("class", axis=1)
y = data["class"]

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de Regresión Logística One-vs-All
model = LogisticRegression(multi_class='ovr', max_iter=200, solver='lbfgs')
model.fit(X_train, y_train)

# Predecir las etiquetas del conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión (accuracy)
accuracy = accuracy_score(y_test, y_pred)

# Calcular el f1-score
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
