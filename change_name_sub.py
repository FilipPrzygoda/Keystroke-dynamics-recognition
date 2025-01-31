import os
import pandas as pd

# Ścieżka do folderu "dane"
folder_path = "dane"

# Wartość startowa dla kolumny "subject"
current_subject = 100

# Iteruj przez wszystkie pliki w folderze
for file_name in os.listdir(folder_path):
    # Sprawdź, czy plik kończy się na "_.tie5Roanl_keystroke_aggregated.csv"
    if file_name.endswith("_.tie5Roanl_keystroke_aggregated.csv"):
        file_path = os.path.join(folder_path, file_name)
        
        # Wczytaj plik CSV
        df = pd.read_csv(file_path)
        
        # Ustaw tę samą wartość "subject" dla całego pliku
        df['subject'] = current_subject
        
        # Zapisz zmodyfikowany plik CSV
        df.to_csv(file_path, index=False)

        print(f"Zaktualizowano kolumnę 'subject' w pliku: {file_name} na wartość: {current_subject}")
        
        # Zwiększ wartość "subject" dla kolejnego pliku
        current_subject += 1