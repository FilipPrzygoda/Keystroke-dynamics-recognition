import csv
import os
import time
from pynput import keyboard

def get_user_id():
    user_id = input("Podaj nazwę użytkownika: ")
    return user_id

def get_password():
    password = input("Podaj hasło użytkownika: ")
    return password

# ------------------------------------------------------------
#                  KONFIGURACJA I STAŁE
# ------------------------------------------------------------
USER_ID = get_user_id()
PASSWORD = get_password()  # Zakładamy, że użytkownik wpisuje np. "abc"
TIMEOUT_DURATION = 30 * 60  # 30 minut w sekundach

FOLDER_NAME = "dane"
FILE_NAME_1 = os.path.join(FOLDER_NAME, f"{USER_ID}_{PASSWORD}_keystroke_aggregated.csv")  
FILE_NAME_2 = os.path.join(FOLDER_NAME, f"{USER_ID}_{PASSWORD}_keystroke_raw.csv")
SESSION_FILE = os.path.join(FOLDER_NAME, f"{USER_ID}_{PASSWORD}_last_session_id.txt")
PASSWORD = PASSWORD + '\n'
if not os.path.exists(FOLDER_NAME):
    os.makedirs(FOLDER_NAME)

# ------------------------------------------------------------
#                  ZMIENNE GLOBALNE
# ------------------------------------------------------------
session_id = 1        # Numer sesji
rep_number = 0        # Numer powtórki (rep) wewnątrz danej sesji
characters_count = 0
last_activity_time = time.time()
user_input = ""

# --- Zmienne do logowania "raw" (file_2) ---
press_times = {}       # (key_char, press_time)
release_times = {}

# --- Zmienne do logowania "aggregated" (file_1) ---
# Będziemy przechowywać czasy w strukturach pomocniczych.
# Index w tablicach = index znaku w haśle.
pw = PASSWORD         # Dla wygody
pw_length = len(pw)
hold_times = [0.0] * pw_length     # H.x
dd_times   = [0.0] * (pw_length - 1) if pw_length > 1 else []
ud_times   = [0.0] * (pw_length - 1) if pw_length > 1 else []

# next_char_idx – wskaźnik na kolejny znak hasła, który powinien zostać wciśnięty,
# aby uznać, że faktycznie użytkownik pisze to hasło w prawidłowej kolejności.
next_char_idx = 0


previous_press_time = None
previous_release_time = None
previous_key_char = None

# <-- NOWE: Śledzenie klawiszy modyfikujących -->
shift_pressed = False
altgr_pressed = False

# ------------------------------------------------------------
#                  FUNKCJE POMOCNICZE
# ------------------------------------------------------------
def initialize_session_id():
    """
    Funkcja inicjalizująca zmienną session_id i zapisująca ją do pliku.
    Jeśli plik z poprzednim session_id istnieje, zwiększamy wartość o 1.
    """
    global session_id
    if os.path.isfile(SESSION_FILE):
        with open(SESSION_FILE, mode='r', encoding='utf-8') as f:
            last_session_id = f.read()
            try:
                session_id = int(last_session_id) + 1
            except ValueError:
                session_id = 1
    else:
        session_id = 1
    characters_count = 0
    save_session_id()

def save_session_id():
    with open(SESSION_FILE, mode='w', encoding='utf-8') as f:
        f.write(str(session_id))

def initialize_csv_file_1():
    """
    Inicjalizacja pliku file_1, w którym zapisujemy dane zagregowane
    (jedna próba = jeden wiersz).
    
    Pierwsze trzy kolumny to: subject, sessionIndex, rep
    Następne kolumny to: H.x, DD.x.y, UD.x.y
    – generowane dynamicznie w zależności od treści hasła PASSWORD.
    """
    if not os.path.isfile(FILE_NAME_1):
        with open(FILE_NAME_1, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ["subject", "sessionIndex", "rep"]

            # Generujemy dynamiczne kolumny w zależności od hasła
            # Przykład dla hasła "abc":
            #  -> H.a, DD.a.b, UD.a.b, H.b, DD.b.c, UD.b.c, H.c
            next_ch = pw[0]
            if next_ch == '\n':
                next_ch = "Return"
            elif next_ch == '.':
                next_ch = "period"
            elif next_ch == '5':
                next_ch = "five"
            elif next_ch == 'R':
                next_ch = "Shift.r"
            for i in range(pw_length):
                # H.x
                ch = next_ch
                header.append(f"H.{ch}")
                # Sprawdzamy czy jest kolejny znak, wtedy dodajemy DD i UD
                if i < pw_length - 1:
                    next_ch = pw[i+1]
                    if next_ch == '\n':
                        next_ch = "Return"
                    elif next_ch == '.':
                        next_ch = "period"
                    elif next_ch == '5':
                        next_ch = "five"
                    elif next_ch == 'R':
                        next_ch = "Shift.r"
                    header.append(f"DD.{ch}.{next_ch}")
                    header.append(f"UD.{ch}.{next_ch}")

            writer.writerow(header)

def initialize_csv_file_2():
    """
    Inicjalizacja pliku file_2, w którym zapisujemy surowe dane
    (podobnie jak w dotychczasowym kodzie).
    """
    if not os.path.isfile(FILE_NAME_2):
        with open(FILE_NAME_2, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "User_ID","Session_ID", 
                "rep", 
                "Key_Pressed","Key_Pressed_Previous",
                "Press_Time", "Release_Time", 
                "Hold_Time","DD","UD", 
                "Characters_Count"  # opcjonalnie
            ])

def reset_aggregated_arrays():
    """
    Resetuje tablice z czasami (H, DD, UD) do zera
    oraz indeks znaku (next_char_idx).
    """
    global hold_times, dd_times, ud_times
    global next_char_idx, previous_release_time
    global user_input
    hold_times = [0.0] * pw_length
    dd_times   = [0.0] * (pw_length - 1) if pw_length > 1 else []
    ud_times   = [0.0] * (pw_length - 1) if pw_length > 1 else []
    next_char_idx = 0
    previous_release_time = None
    user_input =""

def save_aggregated_row():
    """
    Zapisuje do file_1 JEDEN wiersz z czasami dla całej próby wpisania hasła.
    """
    with open(FILE_NAME_1, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        row = [USER_ID, session_id, rep_number]

        # Uzupełniamy: H.x, DD.x.y, UD.x.y
        # Dokładnie w tej samej kolejności, jak zdefiniowaliśmy w nagłówku.
        idx_dd_ud = 0
        for i in range(pw_length):
            # H.x
            row.append(f"{hold_times[i]:.4f}")  # formatowanie do 4 miejsc po przecinku
            # Sprawdzamy czy jest kolejny znak
            if i < pw_length - 1:
                row.append(f"{dd_times[i]:.4f}") 
                row.append(f"{ud_times[i]:.4f}") 

        writer.writerow(row)

def save_raw_data(data):
    """
    Zapisuje jeden wiersz do pliku file_2 z surowymi danymi.
    """
    with open(FILE_NAME_2, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(data)


# Funkcje obsługujące zdarzenia klawiszy
def on_press(key):
    global last_activity_time, shift_pressed
    current_time = time.time()
    last_activity_time = current_time
    try:
        key_char = key.char
    except AttributeError:
        key_char = str(key)

    # Jeśli klawisz to SHIFT -> zapamiętujemy
    if key in [keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r]:
        shift_pressed = True


    # Generowanie unikalnego identyfikatora klawisza
    key_id = (key_char, current_time)
    # Zapis czasu naciśnięcia klawisza
    press_times[key_id] = current_time

def on_release(key):
    global next_char_idx, rep_number
    global characters_count, last_activity_time, altgr_pressed
    global previous_press_time, previous_release_time, previous_key_char, previous_key_id
    global user_input, shift_pressed
    current_time = time.time()
    last_activity_time = current_time
    
    # Jeśli puszczamy SHIFT -> kasujemy jego flagę
    if key in [keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r]:
        shift_pressed = False


    try:
        key_char = key.char
    except AttributeError:
        key_char = str(key)
    # Znalezienie odpowiedniego key_id
    key_ids = [k for k in press_times.keys() if k[0] == key_char]
    if key_ids:
        key_id = key_ids[0]
    else:
        # Jeśli nie znaleziono, tworzymy nowy wpis
        key_id = (key_char, current_time)
        press_times[key_id] = current_time
    press_time = press_times.get(key_id, current_time)
    hold_time = current_time - press_time  # Hold_Time

    release_times[key_id] = current_time

    # Obliczanie Down_Down_Time (DD) i Up_Down_Time (UD)
    if previous_press_time is not None:
        DD = press_time - previous_press_time
    else:
        DD = None
    if previous_release_time is not None:
        UD = press_time - previous_release_time
    else:
        UD = None

    # Jeśli chcesz ręcznie wymuszać wielką literę, gdy SHIFT jest wciśnięty:
    if shift_pressed:
        key_char = key_char.upper()

    # Inkrementacja characters_count przed zapisem
    characters_count += 1

    # Zapis do surowego pliku (file_2):
    data1 = [
        USER_ID,
        session_id,
        rep_number,
        key_char,
        previous_key_char,
        press_time,
        current_time,
        hold_time,
        DD,
        UD,
        characters_count
    ]
    save_raw_data(data1)


    # Aktualizacja poprzednich wartości
    previous_key_char = key_char
    previous_press_time = press_time
    previous_release_time = current_time
    
    if key_char == "Key.ctrl_l":
        print(key_char)
    if key_char == "Key.ctrl_l" or key_char == "Key.alt_gr" or key_char == "Key.shift" or key_char == "Key.alt_l":
        key_char = ""
    elif key_char == "Key.enter":
        key_char = '\n'

    if key_char == "Key.backspace":
        user_input = user_input[:-1]
    else:
        user_input += key_char
        if user_input == pw[:next_char_idx+1]:
            # Zapisujemy hold_time
            hold_times[next_char_idx] = hold_time

            # Zapisujemy dd/ud w tablicach, jeśli to nie jest pierwszy znak
            if next_char_idx > 0:
                dd_times[next_char_idx - 1] = press_time - globals()['previous_press_t'] if globals()['previous_press_t'] else 0.0
                ud_times[next_char_idx - 1] = press_time - previous_release_time if previous_release_time else 0.0

            next_char_idx += 1

            # Jeśli doszliśmy do końca hasła, to uznajemy próbę (rep) za zakończoną
            if next_char_idx == pw_length:
                rep_number += 1
                # Zapisujemy wiersz do file_1
                print("zapisano")
                save_aggregated_row()
                # Resetujemy tablice i indeksy, by móc zarejestrować kolejną próbę
                reset_aggregated_arrays()
        elif key_char == '\n':
            print("cos zle")
            reset_aggregated_arrays()
# ----------------------------------------------------
    #  Uaktualniamy previous_* do obliczeń dd/ud przy
    #  następnym naciśnięciu
    # ----------------------------------------------------
    print(user_input)
    globals()['previous_press_t'] = previous_press_time
    previous_release_time = previous_release_time

    # Aktualizacja previous_key_id na obecny klawisz
    previous_key_id = key_id
    for k in key_ids:
        del press_times[k]

# ------------------------------------------------------------
#                  INICJALIZACJA PLIKÓW I SESJI
# ------------------------------------------------------------
initialize_session_id()
initialize_csv_file_1()
initialize_csv_file_2()

# ------------------------------------------------------------
#                  GŁÓWNA PĘTLA
# ------------------------------------------------------------
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    try:
        while True:
            time.sleep(1)
            # Sprawdzenie, czy upłynęło 30 minut bezczynności
            if time.time() - last_activity_time > TIMEOUT_DURATION:
                # Zamykamy poprzednią sesję i rozpoczynamy nową
                session_id += 1
                save_session_id()

                # Reset rep
                rep_number = 0
                
                # Reset zmiennych
                user_input = ""
                previous_press_time = None
                previous_release_time = None
                previous_key_char = None
                reset_aggregated_arrays()
                press_times.clear()
                release_times.clear()
                globals()['previous_press_t'] = None
                previous_release_time = None

                last_activity_time = time.time()

    except KeyboardInterrupt:
        pass
