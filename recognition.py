import numpy as np
import pickle
import tensorflow as tf
from pynput import keyboard
from tensorflow.keras.models import load_model

###############################################################################
# 1. Stałe / Ścieżki do plików
###############################################################################
REFERENCE_PASSWORD = ".tie5Roanl"  # Hasło, które użytkownik ma wprowadzić

MODEL_PATH = "lstm_model_tf.h5"       # Wytrenowany model
SCALER_PATH = "scaler.pkl"            # StandardScaler (pickle)
LABEL_ENCODER_PATH = "label_encoder.pkl"

SESSION_INDEX = 1     # Stałe "sessionIndex"
rep_counter = 1       # rep, rosnący z każdą próbą wpisania hasła

###############################################################################
# 2. Ładowanie modelu, scalera, LabelEncoder
###############################################################################
print(f"Ładowanie modelu z: {MODEL_PATH}")
model = load_model(MODEL_PATH)

print(f"Ładowanie scalera z: {SCALER_PATH}")
with open(SCALER_PATH, "rb") as scf:
    scaler = pickle.load(scf)

print(f"Ładowanie LabelEncoder z: {LABEL_ENCODER_PATH}")
with open(LABEL_ENCODER_PATH, "rb") as lef:
    label_encoder = pickle.load(lef)

###############################################################################
# 3. Zmienne pomocnicze do nasłuchiwania
###############################################################################
typed_string = []    # przechowuje wpisywane znaki (dla kontroli/porównania)
typed_events = []    # przechowuje tuple: (char, press_time, release_time)
pressed_keys = {}    # mapuje klawisz -> czas_naciśnięcia


###############################################################################
# 4. Funkcja compute_features: Zwraca 33 cechy (sessionIndex, rep, H, DD, UD...)
###############################################################################
def compute_features(events, session_idx, rep):
    """
    Zwraca wektor 33 cech w kolejności:
      [ sessionIndex, rep,
        H.period, DD.period.t, UD.period.t,
        H.t, DD.t.i, UD.t.i,
        ...
        H.Return
      ]
    Musisz wypełnić logikę liczenia H, DD, UD zgodnie z plikiem DSL-StrongPasswordData.csv.
    
    Parametry:
    -----------
    events: list[(char, press_time, release_time)]
        np. [('.', 1.234, 1.450), ('t', 1.501, 1.620), ...]
    session_idx: int
        Wartość kolumny sessionIndex
    rep: int
        Wartość kolumny rep

    Zwraca:
    -----------
    feature_vector: list[float]
        Dokładnie 33 wartości float w odpowiedniej kolejności.
    """
    
    # Przygotuj wektor 33 cech:
    feature_vector = [0.0] * 33

    # 1) sessionIndex i rep w pierwszych dwóch pozycjach
    feature_vector[0] = float(session_idx)
    feature_vector[1] = float(rep)

    # 2) Oblicz H, DD, UD dla poszczególnych par znaków w TAKIEJ SAMEJ KOLEJNOŚCI,
    #    w jakiej występują w DSL-StrongPasswordData.csv:
    #
    #    0:  subject (pomijamy w predykcji)
    #    1:  sessionIndex  -> feature_vector[0]
    #    2:  rep           -> feature_vector[1]
    #    3:  H.period      -> feature_vector[2]
    #    4:  DD.period.t   -> feature_vector[3]
    #    5:  UD.period.t   -> feature_vector[4]
    #    6:  H.t           -> feature_vector[5]
    #    7:  DD.t.i        -> feature_vector[6]
    #    8:  UD.t.i        -> feature_vector[7]
    #    ... 
    #    (aż do) 
    #    34: H.Return      -> feature_vector[33 - 1 = 32]
    #
    # Poniżej jedynie demonstracyjny przykład dla paru znaków:
    
    # Twórz słownik {char: (press_time, release_time)} lub klucz to kolejność 
    # Zwróć uwagę, że SHIFT, Return i kropka '.' też musisz rejestrować!
    # W DSL jest np.: . (kropka), t, i, e, five, Shift.r, o, a, n, l, Return
    
    # Najpierw zrób mapę znak -> (press, release) lub listę:
    events_dict = {}
    for (ch, p, r) in events:
        # klucz tekstowy, np. 't', '.', 'Key.enter' (może SHIFT)
        # W DSL SHIFT występuje jako 'Shift.r' w kolumnach. 
        # W Twoim realnym nasłuchu to może być str(key) == 'Key.shift_r' itp.
        events_dict[ch] = (p, r)
    
    # Przykład: H.period = release('.') - press('.')
    #           Indeks kolumny w feature_vector to 2
    if '.' in events_dict:
        p_dot, r_dot = events_dict['.']
        feature_vector[2] = r_dot - p_dot  # H.period
    
    # Przykład: DD.period.t = press('t') - press('.')
    #           Indeks kolumny w feature_vector to 3
    if '.' in events_dict and 't' in events_dict:
        p_t, r_t = events_dict['t']
        p_dot, r_dot = events_dict['.']
        feature_vector[3] = p_t - p_dot  # DD.period.t

    # Przykład: UD.period.t = press('t') - release('.')
    #           Indeks kolumny w feature_vector to 4
    if '.' in events_dict and 't' in events_dict:
        p_t, r_t = events_dict['t']
        p_dot, r_dot = events_dict['.']
        feature_vector[4] = p_t - r_dot  # UD.period.t

    # ... i tak dalej, analogicznie do pliku DSL ...
    #
    # Pamiętaj o SHIFT, np. DSL ma kolumny DD.five.Shift.r, UD.five.Shift.r, H.Shift.r...
    # Key.enter -> 'Return' w DSL.
    #
    # feature_vector[...] = ...
    
    return feature_vector


###############################################################################
# 5. Funkcja klasyfikująca
###############################################################################
def classify_typed_password(session_idx, rep):
    # 1) Wyliczenie cech
    features = compute_features(typed_events, session_idx, rep)

    # 2) Transformacja do kształtu (1, 33)
    X_input = np.array(features).reshape(1, -1)

    # 3) Skalowanie
    X_scaled = scaler.transform(X_input)

    # 4) Dla LSTM: (1, 1, 33)
    X_seq = np.expand_dims(X_scaled, axis=1)

    # 5) Predykcja
    y_pred = model.predict(X_seq)
    pred_class = np.argmax(y_pred, axis=1)
    subject_name = label_encoder.inverse_transform(pred_class)
    print(">>> Rozpoznany subject to:", subject_name[0])


###############################################################################
# 6. Obsługa klawiszy - on_press, on_release
###############################################################################
def on_press(key):
    press_time = tf.timestamp().numpy()
    # staramy się wziąć key.char, jeśli to znak
    if hasattr(key, 'char') and key.char is not None:
        pressed_keys[key.char] = press_time
    else:
        # np. SHIFT, ENTER
        pressed_keys[str(key)] = press_time

def on_release(key):
    global rep_counter  # chcemy modyfikować globalną zmienną

    release_time = tf.timestamp().numpy()

    # Zwykły znak
    if hasattr(key, 'char') and key.char is not None:
        ch = key.char
        if ch in pressed_keys:
            p_time = pressed_keys[ch]
            typed_events.append((ch, p_time, release_time))
            pressed_keys.pop(ch, None)
        typed_string.append(ch)
        print("".join(typed_string), end="\r")  # podgląd (odświeżanie w miejscu)

    else:
        # Klawisz specjalny
        ch = str(key)
        if ch in pressed_keys:
            p_time = pressed_keys[ch]
            typed_events.append((ch, p_time, release_time))
            pressed_keys.pop(ch, None)

        # ENTER => sprawdzamy hasło
        if key == keyboard.Key.enter:
            typed_text = "".join(typed_string)
            print("\nWpisane hasło to:", typed_text)

            if typed_text == REFERENCE_PASSWORD:
                print("Hasło się zgadza. Rozpoznaję użytkownika...")

                # Wywołanie klasyfikacji z sessionIndex=1 i rep=rep_counter
                classify_typed_password(session_idx=SESSION_INDEX, rep=rep_counter)

                # Zwiększamy rep_counter
                rep_counter += 1
            else:
                print("Błędne hasło. Nie rozpoznaję.")

            # Czyścimy typed_events i typed_string
            typed_events.clear()
            typed_string.clear()

        elif key == keyboard.Key.backspace:
            # Usunięcie ostatniego znaku (jeśli jest)
            if typed_string:
                typed_string.pop()
            print("".join(typed_string), end="\r")


###############################################################################
# 7. Uruchomienie nasłuchiwania
###############################################################################
def main():
    from pynput import keyboard
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        print(f"Rozpoczynam nasłuch klawiatury.\n"
              f"Wpisz hasło: {REFERENCE_PASSWORD} i naciśnij ENTER.\n"
              f"sessionIndex = {SESSION_INDEX}, kolejne rep od 1 wzwyż.\n")
        listener.join()

if __name__ == "__main__":
    main()
