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
# 3. Funkcje do wyznaczania FAR, FRR i EER
#    (replikujemy tu compute_frr_far_eer, bo jest używana przez compute_average_frr_far_eer)
###############################################################################
def compute_frr_far_eer(model, X, y, user_index, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100, endpoint=True)

    preds = model.predict(X)
    user_probs = preds[:, user_index]

    genuine_mask = (y == user_index)
    impostor_mask = (y != user_index)

    total_genuine = np.sum(genuine_mask)
    total_impostor = np.sum(impostor_mask)

    frr_list = []
    far_list = []

    for t in thresholds:
        accepted_genuine = np.sum(user_probs[genuine_mask] >= t)
        rejected_genuine = total_genuine - accepted_genuine
        FRR = rejected_genuine / total_genuine if total_genuine > 0 else 0.0

        accepted_impostor = np.sum(user_probs[impostor_mask] >= t)
        FAR = accepted_impostor / total_impostor if total_impostor > 0 else 0.0

        frr_list.append(FRR)
        far_list.append(FAR)

    frr_arr = np.array(frr_list)
    far_arr = np.array(far_list)

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
# 4. Wyliczanie i rysowanie ŚREDNIEGO FAR/FRR/EER dla całej bazy
###############################################################################
def main_evaluation_average():
    global X_test_seq, y_test

    thresholds, avg_frr, avg_far, eer_all, eer_threshold_all = compute_average_frr_far_eer(
        lstm_model, X_test_seq, y_test
    )
    eer_index = np.argmin(np.abs(avg_frr - avg_far))

    print(f"\nŚREDNI EER DLA CAŁEJ BAZY = {eer_all:.4f}")
    print(f"AVG_FRR = {avg_frr[eer_index]:.4f}, AVG_FAR = {avg_far[eer_index]:.4f}, próg = {eer_threshold_all:.4f}")

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
# Wywołanie głównej funkcji
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main_evaluation_average()
