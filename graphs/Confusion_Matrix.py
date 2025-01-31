import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
import pickle

###############################################################################
# Co to jest macierz pomyłek?
###############################################################################
# Macierz pomyłek (Confusion Matrix) to tablica, która przedstawia wyniki
# klasyfikacji modelu. Jest to narzędzie umożliwiające ocenę dokładności
# modelu, pokazując liczby prawidłowych i błędnych klasyfikacji
# dla każdej klasy. 
# 
# Komórki na przekątnej macierzy reprezentują prawidłowe klasyfikacje,
# a poza przekątną - błędne klasyfikacje.
###############################################################################

# Ładowanie modelu i etykiet (LabelEncoder)
MODEL_PATH = "lstm_model_tf.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"
X_TEST_SEQ_PATH = "X_test_seq.npy"
Y_TEST_PATH = "y_test.npy"

print("\u0141adowanie modelu z pliku:", MODEL_PATH)
lstm_model = load_model(MODEL_PATH)

print("\u0141adowanie LabelEncoder z pliku:", LABEL_ENCODER_PATH)
with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

# Ładowanie danych testowych
X_test_seq = np.load(X_TEST_SEQ_PATH)
y_test = np.load(Y_TEST_PATH)

print("Kształt danych testowych:")
print("X_test_seq:", X_test_seq.shape)
print("y_test:", y_test.shape)

# Generowanie predykcji modelu
y_pred_proba = lstm_model.predict(X_test_seq)  # Prawdopodobieństwa dla każdej klasy
y_pred = np.argmax(y_pred_proba, axis=1)  # Klasa z najwyższym prawdopodobieństwem

# Obliczanie macierzy pomyłek
cm = confusion_matrix(y_test, y_pred)
class_names = le.classes_  # Pobranie nazw klas z LabelEncoder

# Wizualizacja macierzy pomyłek
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45)  # Obrót etykiet na osi X
plt.yticks(rotation=0)   # Obrót etykiet na osi Y
plt.tight_layout()
plt.show()
