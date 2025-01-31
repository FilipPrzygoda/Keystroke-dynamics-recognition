import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 1. Wczytaj dane
data = pd.read_csv("DSL-StrongPasswordData.csv")

# 2. Przetwarzanie danych
# Wybierz cechy (feature columns)
feature_columns = [
    'H.period', 'DD.period.t', 'UD.period.t', 'H.t', 'DD.t.i', 'UD.t.i',
    'H.i', 'DD.i.e', 'UD.i.e', 'H.e', 'DD.e.five', 'UD.e.five',
    'H.five', 'DD.five.Shift.r', 'UD.five.Shift.r', 'H.Shift.r',
    'DD.Shift.r.o', 'UD.Shift.r.o', 'H.o', 'DD.o.a', 'UD.o.a',
    'H.a', 'DD.a.n', 'UD.a.n', 'H.n', 'DD.n.l', 'UD.n.l', 'H.l',
    'DD.l.Return', 'UD.l.Return', 'H.Return'
]

X = data[feature_columns]

# Standaryzacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Klasteryzacja (k-means)
n_clusters = 2  # Liczba emocji
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['label'] = kmeans.fit_predict(X_scaled)

# 4. Podział danych na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X_scaled, data['label'], test_size=0.2, random_state=42)

# One-hot encoding dla etykiet
y_train_categorical = pd.get_dummies(y_train).values
y_test_categorical = pd.get_dummies(y_test).values

# 5. Budowa modelu
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.25),
    Dense(512, activation='relu'),
    Dropout(0.25),
    Dense(n_clusters, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Trenowanie modelu
history = model.fit(
    X_train, y_train_categorical,
    validation_split=0.2,
    epochs=70,
    batch_size=1024,
    verbose=1
)

# 7. Ocena modelu
loss, accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)
print(f"Dokładność modelu: {accuracy * 100:.2f}%")

# 8. Wykresy wyników
# Strata i dokładność podczas treningu
plt.figure(figsize=(12, 5))

# Strata
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Strata treningowa')
plt.plot(history.history['val_loss'], label='Strata walidacyjna')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()
plt.title('Strata podczas treningu')

# Dokładność
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Dokładność treningowa')
plt.plot(history.history['val_accuracy'], label='Dokładność walidacyjna')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()
plt.title('Dokładność podczas treningu')

plt.show()

# 9. Prognozowanie na nowych danych
# Przykład prognozy dla pierwszych 5 próbek z testowego zbioru danych
predictions = model.predict(X_test[:5])
predicted_classes = np.argmax(predictions, axis=1)
print("Prognozowane klasy:", predicted_classes)
print("Rzeczywiste klasy:", y_test[:5].values)

from sklearn.metrics import silhouette_score

# Po wykonaniu klasteryzacji k-means
silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)

# Wyświetlenie wyniku
print(f"Silhouette Score: {silhouette_avg:.2f}")

# Opcjonalnie: Warunki interpretacji wyniku
if silhouette_avg > 0.5:
    print("Klastery są dobrze rozdzielone.")
elif silhouette_avg > 0:
    print("Klastery częściowo się nakładają.")
else:
    print("Punkty danych mogą być przypisane do niewłaściwych klastrów.")

model_path = "emotions_model_tf.h5"
model.save(model_path)
