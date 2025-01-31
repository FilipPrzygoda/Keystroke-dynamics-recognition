import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

###############################################################################
# 1. Ładowanie wytrenowanego modelu i obiektu LabelEncoder
###############################################################################
MODEL_PATH = "lstm_model_tf.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"
X_TEST_SEQ_PATH = "X_test_seq.npy"
Y_TEST_PATH = "y_test.npy"

print(f"Ładowanie modelu z pliku: {MODEL_PATH}")
lstm_model = load_model(MODEL_PATH)

print(f"Ładowanie LabelEncoder z pliku: {LABEL_ENCODER_PATH}")
with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

###############################################################################
# 2. Wczytanie danych testowych (X_test_seq, y_test)
#    Jeśli potrzebujesz dodatkowych kroków (skalowanie itp.), tutaj je zaimplementuj.
###############################################################################
X_test_seq = np.load(X_TEST_SEQ_PATH)
y_test = np.load(Y_TEST_PATH)

print("Kształt X_test_seq:", X_test_seq.shape)
print("Kształt y_test:", y_test.shape)

###############################################################################
# 3. Funkcja do wyznaczania FAR, FRR i EER dla pojedynczego użytkownika
###############################################################################
def compute_frr_far_eer(model, X, y, user_index, thresholds=None):
    """
    Oblicza FRR, FAR oraz EER (oraz próg EER) dla pojedynczego użytkownika (user_index).

    Parametry:
    -----------
    model : tf.keras.Model
        Załadowany/wytrenowany model (np. LSTM).
    X : np.ndarray
        Dane testowe (sekwencje) o kształcie (liczba_próbek, sekwencja, liczba_cech).
    y : np.ndarray
        Etykiety testowe (LabelEncoder).
    user_index : int
        Indeks użytkownika (np. 0, 1, 2, ...) dla którego liczymy FRR/FAR.
    thresholds : np.ndarray (opcjonalnie)
        Tablica progów decyzyjnych w zakresie [0, 1]. Jeśli None, tworzymy 100-punktową.

    Zwraca:
    -----------
    thresholds : np.ndarray
        Użyte progi.
    frr_arr : np.ndarray
        Wartości FRR dla kolejnych progów.
    far_arr : np.ndarray
        Wartości FAR dla kolejnych progów.
    eer : float
        Equal Error Rate (wartość, przy której FAR = FRR).
    eer_threshold : float
        Próg decyzyjny, przy którym osiągamy EER.
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100, endpoint=True)

    # Predykcje modelu (prawdopodobieństwa dla każdej klasy)
    preds = model.predict(X)  # (liczba_próbek, liczba_klas)

    # Prawdopodobieństwo, że próbka należy do user_index
    user_probs = preds[:, user_index]

    # Maska dla próbek user_index (genuine) i pozostałych (impostors)
    genuine_mask = (y == user_index)
    impostor_mask = (y != user_index)

    total_genuine = np.sum(genuine_mask)
    total_impostor = np.sum(impostor_mask)

    frr_list = []
    far_list = []

    for t in thresholds:
        # Akceptujemy, jeśli prawdopodobieństwo >= t
        accepted_genuine = np.sum(user_probs[genuine_mask] >= t)
        rejected_genuine = total_genuine - accepted_genuine
        FRR = rejected_genuine / total_genuine if total_genuine > 0 else 0.0

        accepted_impostor = np.sum(user_probs[impostor_mask] >= t)
        FAR = accepted_impostor / total_impostor if total_impostor > 0 else 0.0

        frr_list.append(FRR)
        far_list.append(FAR)

    frr_arr = np.array(frr_list)
    far_arr = np.array(far_list)

    # Szukamy punktu EER - minimalna różnica |FAR - FRR|
    diff = np.abs(frr_arr - far_arr)
    min_index = np.argmin(diff)

    eer = (frr_arr[min_index] + far_arr[min_index]) / 2.0
    eer_threshold = thresholds[min_index]

    return thresholds, frr_arr, far_arr, eer, eer_threshold

###############################################################################
# 4. Wyliczanie i rysowanie FAR/FRR/EER dla zadanego użytkownika
###############################################################################
def main_evaluation_for_user():
    global X_test_seq, y_test, le

    # Zmień poniższą wartość na dowolnego istniejącego użytkownika z bazy.
    # Uwaga: to musi być etykieta w oryginalnej postaci (np. "Julia21k", 101 itp.),
    # aby LabelEncoder mógł ją przekształcić.
    wybrany_uzytkownik = 101

    # Sprawdzamy, czy LabelEncoder zna taką etykietę
    try:
        user_index = le.transform([wybrany_uzytkownik])[0]
    except ValueError:
        print(f"Użytkownik {wybrany_uzytkownik} nie występuje w LabelEncoder.")
        return

    thresholds, frr_arr, far_arr, eer_user, eer_threshold_user = compute_frr_far_eer(
        lstm_model, X_test_seq, y_test, user_index
    )

    # Znajdujemy index, w którym FAR i FRR są najbliżej
    min_idx = np.argmin(np.abs(frr_arr - far_arr))
    print(f"\nWybrany użytkownik: {wybrany_uzytkownik} (indeks={user_index})")
    print(f"EER = {eer_user:.4f}, FRR = {frr_arr[min_idx]:.4f}, FAR = {far_arr[min_idx]:.4f} przy progu = {eer_threshold_user:.4f}")

    # Wykres FAR/FRR
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, frr_arr, label="FRR", color="blue")
    plt.plot(thresholds, far_arr, label="FAR", color="red")
    plt.axvline(eer_threshold_user, color="green", linestyle="--", label="EER Threshold")
    plt.title(f"FAR/FRR - Użytkownik: {wybrany_uzytkownik}")
    plt.xlabel("Próg (threshold)")
    plt.ylabel("Wartość współczynnika")
    plt.legend()
    plt.grid(True)
    plt.show()

# ---------------------------------------------------------------------------
# Wywołanie głównej funkcji
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main_evaluation_for_user()
