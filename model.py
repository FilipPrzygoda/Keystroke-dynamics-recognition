import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Zmienna określająca, czy chcemy wczytać główny plik
use_main_data = True

# Zmienna określająca, czy chcemy wczytać pliki z folderu "dane"
use_additional_data = False

# Ścieżka do głównego pliku
data_path = "DSL-StrongPasswordData.csv"

# Ścieżka do folderu "dane" oraz hasło (część wzorca w nazwie pliku)
folder_path = "dane"
password = ".tie5Roanl"

# Lista DataFrame'ów ze wszystkich źródeł, którą na końcu połączymy
df_list = []

# Sprawdzenie dostępności GPU
print("Dostępne urządzenia TensorFlow:")
print(tf.config.list_physical_devices())

# Wymuszenie użycia GPU, jeśli jest dostępny
if tf.config.list_physical_devices('GPU'):
    print("\nTensorFlow będzie używać GPU.")
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
else:
    print("\nTensorFlow NIE używa GPU. Sprawdź instalację sterowników CUDA i cuDNN.")

# =====================================================================
# 1. Wczytanie danych z pliku DSL-StrongPasswordData.csv (opcjonalnie)
# =====================================================================
if use_main_data:
    try:
        df_main = pd.read_csv(data_path)
        df_list.append(df_main)
        print(f"Wczytano główny plik: {data_path}, rozmiar: {df_main.shape}")
    except FileNotFoundError:
        print(f"Nie znaleziono pliku: {data_path} (use_main_data=True)")

# =====================================================================
# 2. Wczytanie danych z folderu 'dane' (opcjonalnie)
# =====================================================================
if use_additional_data:
    pattern = os.path.join(folder_path, f"*_{password}_keystroke_aggregated.csv")
    file_list = glob.glob(pattern)

    for file in file_list:
        try:
            temp_df = pd.read_csv(file)
            df_list.append(temp_df)
            print(f"Wczytano plik: {file}, rozmiar: {temp_df.shape}")
        except Exception as e:
            print(f"Nie udało się wczytać pliku {file}. Błąd: {e}")

# =====================================================================
# 3. Łączenie danych w jeden DataFrame
# =====================================================================
if df_list:
    df = pd.concat(df_list, ignore_index=True)
    print(f"\nPołączony DataFrame ma kształt: {df.shape}")
else:
    # Jeśli nie wczytano żadnych plików (puste df_list),
    # można wywołać wyjątek lub przypisać None
    df = None
    print("\nNie wczytano żadnych danych - df = None")
    

# =====================================================================
# 4. Dalsze przetwarzanie
# =====================================================================
# Przykład, gdy df nie jest None i chcemy kontynuować dalsze kroki:
if df is not None:
    label_col = 'subject'
    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols].values
    y = df[label_col].values

    # Dalej można użyć np. LabelEncoder i kontynuować przetwarzanie...

# Przygotowanie cech i etykiet
label_col = 'subject'
feature_cols = [c for c in df.columns if c != label_col]

X = df[feature_cols].values
y = df[label_col].values

# Kodowanie etykiet (jeśli typ etykiet nie jest numeryczny)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(np.unique(y_encoded))

# Zapis LabelEncoder do pliku
label_encoder_path = "label_encoder.pkl"

with open(label_encoder_path, "wb") as le_file:
    pickle.dump(le, le_file)

print(f"LabelEncoder zapisano do pliku: {label_encoder_path}")

# Podział danych na treningowy, walidacyjny i testowy
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Standaryzacja danych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Przekształcenie danych dla LSTM (dodanie wymiaru sekwencji)
X_train_seq = np.expand_dims(X_train_scaled, axis=1)
X_val_seq = np.expand_dims(X_val_scaled, axis=1)
X_test_seq = np.expand_dims(X_test_scaled, axis=1)

# One-hot encoding etykiet
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_val_cat = to_categorical(y_val, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# =====================================================================
# Model LSTM
# =====================================================================
lstm_model = Sequential([
    LSTM(512, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), activation='tanh'),
    Dropout(0.4),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])

lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Trening modelu z uwzględnieniem zbioru walidacyjnego
history = lstm_model.fit(
    X_train_seq,
    y_train_cat,
    validation_data=(X_val_seq, y_val_cat),
    epochs=140,  # Możesz zmienić liczbę epok
    batch_size=2048,
    verbose=1
)

# Ewaluacja modelu na zbiorze testowym
lstm_scores = lstm_model.evaluate(X_test_seq, y_test_cat, verbose=0)
acc_lstm = lstm_scores[1]
print("\nDokładność (accuracy) LSTM na danych testowych:", acc_lstm)

# =====================================================================
# 3. Wizualizacja wyników
# =====================================================================
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Wykres strat
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss - Trening')
    plt.plot(history.history['val_loss'], label='Loss - Walidacja')
    plt.title('Strata podczas treningu')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()

    # Wykres dokładności
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Dokładność - Trening')
    plt.plot(history.history['val_accuracy'], label='Dokładność - Walidacja')
    plt.title('Dokładność podczas treningu')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Zapis modelu
model_path = "lstm_model_tf.h5"
lstm_model.save(model_path)

np.save("X_train_seq.npy", X_train_seq)
np.save("X_val_seq.npy", X_val_seq)
np.save("X_test_seq.npy", X_test_seq)

np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)
np.save("y_test.npy", y_test)
