import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, MeanShift
from sklearn.metrics import accuracy_score, adjusted_rand_score

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

# Dividir los datos en conjunto de entrenamiento y prueba (subconjuntos disjuntos)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K-Means clustering
n_clusters = len(y.unique())
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_train)

# Predecir las etiquetas de los datos de prueba usando K-Means
y_pred_kmeans = kmeans.predict(X_test)

# Evaluar K-Means usando Adjusted Rand Index (ARI)
ari_kmeans = adjusted_rand_score(y_test, y_pred_kmeans)

# Mean-Shift clustering
mean_shift = MeanShift()
mean_shift.fit(X_train)

# Predecir las etiquetas de los datos de prueba usando Mean-Shift
y_pred_meanshift = mean_shift.predict(X_test)

# Evaluar Mean-Shift usando Adjusted Rand Index (ARI)
ari_meanshift = adjusted_rand_score(y_test, y_pred_meanshift)

print(f"K-Means Adjusted Rand Index (ARI): {ari_kmeans:.4f}")
print(f"Mean-Shift Adjusted Rand Index (ARI): {ari_meanshift:.4f}")

# Análisis de los resultados
print("\nAnálisis de resultados:")
print("El ARI mide la similitud entre las etiquetas verdaderas y las etiquetas predichas, ajustando por la probabilidad de asignaciones aleatorias.")
print("Un ARI más alto indica una mejor concordancia entre las etiquetas verdaderas y las etiquetas de clustering.")
print("K-Means tiene un ARI de {:.4f}, mientras que Mean-Shift tiene un ARI de {:.4f}.".format(ari_kmeans, ari_meanshift))
print("Esto sugiere que {}.".format("K-Means se ajusta mejor a los datos" if ari_kmeans > ari_meanshift else "Mean-Shift se ajusta mejor a los datos"))
