import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

###############################################################################
# 1. Ładowanie wytrenowanego modelu i obiektu LabelEncoder
###############################################################################
MODEL_PATH = "lstm_model_tf.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"
X_test_seq = np.load("X_test_seq.npy")
y_test = np.load("y_test.npy")


print(f"Ładowanie modelu z pliku: {MODEL_PATH}")
lstm_model = load_model(MODEL_PATH)

print(f"Ładowanie LabelEncoder z pliku: {LABEL_ENCODER_PATH}")
with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

###############################################################################
# 2. Wczytanie i przygotowanie danych testowych (X_test_seq, y_test)
###############################################################################
# Poniżej jedynie *przykładowa* sekcja. Musisz odtworzyć takie same kroki
# jak w skrypcie, w którym trenowałeś model (ten sam podział i skalowanie).
#
# a) Jeżeli masz pliki CSV lub inny format - wczytaj je
# b) Zastosuj identyczny StandardScaler, co podczas trenowania (musisz go też wczytać)
# c) Zreplikuj transformację do kształtu sekwencji: np. X_test_seq = np.expand_dims(X_test_scaled, axis=1)

# --- PRZYKŁAD (do dostosowania) ---
# 1. Zakładamy, że mamy te same pliki co poprzednio.
#    - Wczytujesz je do DataFrame
#    - Rozdzielasz na cechy i etykiety
#    - Dzielisz na train/val/test (lub wczytujesz gotowy test).
# 2. Skalujesz identycznym scalerem (np. wczytanym z pliku scaler.pkl)
# 3. Rozszerzasz wymiar dla LSTM.

# [Tu jest tylko schemat - dostosuj do siebie!]

# from sklearn.model_selection import train_test_split

# Wczytanie danych (np. z CSV lub sklejonych DataFrame)
# ... wczytaj df_test lub coś podobnego ...

# feature_cols = [c for c in df_test.columns if c != 'subject']
# X_test = df_test[feature_cols].values
# y_test_raw = df_test['subject'].values

# # Zamiana etykiet na liczby
# y_test = le.transform(y_test_raw)

# # Wczytanie tego samego scalera, który został użyty do trenowania
# with open("scaler.pkl", "rb") as fsc:
#     scaler = pickle.load(fsc)

# X_test_scaled = scaler.transform(X_test)

# # Dodanie wymiaru sekwencji dla LSTM
# X_test_seq = np.expand_dims(X_test_scaled, axis=1)

# # Teraz mamy X_test_seq i y_test (numeryczne)

# Dla przykładu załóżmy, że mamy X_test_seq, y_test:
# (Jeżeli już masz X_test_seq i y_test w postaci .npy, możesz je wczytać np.)
# X_test_seq = np.load("X_test_seq.npy")
# y_test = np.load("y_test.npy")

# Sprawdź wymiary:
print("Kształt X_test_seq:", "nieznany (wczytaj swoje dane!)")
print("Kształt y_test:", "nieznany (wczytaj swoje dane!)")

###############################################################################
# 3. Funkcje do wyznaczania FAR, FRR i EER
###############################################################################
def compute_frr_far_eer(model, X, y, user_index, thresholds=None):
    """
    Oblicza FRR, FAR oraz EER (oraz próg EER) dla pojedynczego użytkownika (user_index).
    
    Parametry:
    -----------
    model : tf.keras.Model
        Załadowany/wytrenowany model (np. LSTM).
    X : np.ndarray
        Dane testowe (sekwencje) o kształcie (liczba_próbek, 1, liczba_cech).
    y : np.ndarray
        Etykiety testowe w postaci wartości całkowitych (LabelEncoder).
    user_index : int
        Indeks użytkownika (np. 0, 1, 2, ...) dla którego liczymy FRR, FAR.
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


def compute_average_frr_far_eer(model, X, y, thresholds=None):
    """
    Oblicza uśrednione FRR, FAR oraz EER (oraz próg EER) dla wszystkich użytkowników
    w zbiorze testowym (one-vs-all).
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100, endpoint=True)

    user_indices = np.unique(y)

    frr_matrix = []
    far_matrix = []

    # Liczymy FRR/FAR osobno dla każdego użytkownika
    for user_idx in user_indices:
        _, frr_arr, far_arr, _, _ = compute_frr_far_eer(
            model, X, y, user_idx, thresholds=thresholds
        )
        frr_matrix.append(frr_arr)
        far_matrix.append(far_arr)

    # Średnie wartości FRR i FAR
    frr_matrix = np.array(frr_matrix)  # (liczba_userów, len(thresholds))
    far_matrix = np.array(far_matrix)

    avg_frr = np.mean(frr_matrix, axis=0)
    avg_far = np.mean(far_matrix, axis=0)

    # EER dla uśrednionych krzywych
    diff = np.abs(avg_frr - avg_far)
    min_index = np.argmin(diff)
    eer = (avg_frr[min_index] + avg_far[min_index]) / 2.0
    eer_threshold = thresholds[min_index]

    return thresholds, avg_frr, avg_far, eer, eer_threshold

###############################################################################
# 4. Wyliczanie i rysowanie FAR/FRR/EER
###############################################################################
def main_evaluation():
    # Zakładamy, że X_test_seq i y_test są już dostępne (np. po wczytaniu z pliku).
    # W tej demonstracji kodu przyjmijmy, że "X_test_seq" i "y_test" istnieją
    # jako zmienne globalne lub wczytasz je w sekcji "Wczytanie i przygotowanie danych testowych".
    #
    # KONIECZNIE dostosuj poniższe do miejsca, w którym faktycznie masz te dane.

    global X_test_seq, y_test

    # -- EER dla wybranego użytkownika -----------------------------------------
    wybrany_uzytkownik = 100  # <-- zmień na dowolnego istniejącego w bazie
    try:
        user_index = le.transform([wybrany_uzytkownik])[0]
    except ValueError:
        print(f"Użytkownik {wybrany_uzytkownik} nie występuje w LabelEncoder.")
        return

    thresholds, frr_arr, far_arr, eer_user, eer_threshold_user = compute_frr_far_eer(
        lstm_model, X_test_seq, y_test, user_index
    )

    print(f"\nWybrany użytkownik: {wybrany_uzytkownik} (indeks={user_index})")
    print(f"EER = {eer_user:.4f}, FRR = {frr_arr[np.argmin(np.abs(frr_arr - far_arr))]:.4f}, FAR = {far_arr[np.argmin(np.abs(frr_arr - far_arr))]:.4f} przy progu = {eer_threshold_user:.4f}")

    # Wykres FAR/FRR dla wybranego usera
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, frr_arr, label="FRR", color="blue")
    plt.plot(thresholds, far_arr, label="FAR", color="red")
    plt.axvline(eer_threshold_user, color="green", linestyle="--", label="EER Threshold")
    plt.title(f"FAR/FRR - Użytkownik: {wybrany_uzytkownik} ")
    plt.xlabel("Próg (threshold)")
    plt.ylabel("Wartość współczynnika")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -- Średni EER dla całej bazy --------------------------------------------
    thresholds, avg_frr, avg_far, eer_all, eer_threshold_all = compute_average_frr_far_eer(
        lstm_model, X_test_seq, y_test
    )
    eer_index = np.argmin(np.abs(avg_frr - avg_far))
    print(f"\nŚREDNI EER DLA CAŁEJ BAZY = {eer_all:.4f}, AVG_FRR = {avg_frr[eer_index]:.4f}, AVG_FAR = {avg_far[eer_index]:.4f} przy progu = {eer_threshold_all:.4f}")

    # Wykres uśrednionych krzywych FRR i FAR
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, avg_frr, label="Średnie FRR", color="blue")
    plt.plot(thresholds, avg_far, label="Średnie FAR", color="red")
    plt.axvline(eer_threshold_all, color="green", linestyle="--", label="EER Threshold")
    plt.title("Średnie FAR/FRR - Cała baza")
    plt.xlabel("Próg (threshold)")
    plt.ylabel("Wartość współczynnika")
    plt.legend()
    plt.grid(True)
    plt.show()

# ---------------------------------------------------------------------------
# Wywołanie głównej funkcji (jeśli uruchamiasz ten plik bezpośrednio)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main_evaluation()
