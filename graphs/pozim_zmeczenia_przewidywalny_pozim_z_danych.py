import pandas as pd
import matplotlib.pyplot as plt

#jakie dane trzeba sprawdzic 

# Wczytanie danych z pliku wejściowego
file_path = "DSL-StrongPasswordData.csv"
df = pd.read_csv(file_path)

# Funkcja do obliczenia wartości względem bazowych
def calculate_relative_to_base(user_data):
    base_values = user_data.iloc[0]  # Pierwszy wiersz jako wartość bazowa
    relative_values = user_data / base_values  # Wyliczenie proporcji względem wartości bazowej
    return relative_values

# Grupowanie danych po użytkowniku i obliczanie wartości względem bazowych
df_relative = df.groupby('subject').apply(calculate_relative_to_base)

# Obliczenie średnich wartości dla każdej sesji i powtórzenia
data_mean = df_relative.groupby(['rep', 'sessionIndex']).mean()
data_mean = data_mean.reset_index()

# Zdefiniowanie nowej kolejności kolumn
desired_order = [
    'subject', 'sessionIndex', 'rep', 'H.period', 'DD.period.t', 'UD.period.t',
    'H.t', 'DD.t.i', 'UD.t.i', 'H.i', 'DD.i.e', 'UD.i.e', 'H.e', 'DD.e.five', 'UD.e.five',
    'H.five', 'DD.five.Shift.r', 'UD.five.Shift.r', 'H.Shift.r', 'DD.Shift.r.o', 'UD.Shift.r.o',
    'H.o', 'DD.o.a', 'UD.o.a', 'H.a', 'DD.a.n', 'UD.a.n', 'H.n', 'DD.n.l', 'UD.n.l',
    'H.l', 'DD.l.Return', 'UD.l.Return', 'H.Return'
]
data_mean = data_mean[desired_order]
data_mean = data_mean.sort_values(by=['sessionIndex', 'rep'])

# Dodanie indeksu do śledzenia kolejności wpisania danych
data_mean['index'] = range(1, len(data_mean) + 1)

# Kolumny, które chcesz wizualizować
columns_to_plot = [
    'H.period', 'DD.period.t', 'UD.period.t', 'H.t', 'DD.t.i', 'UD.t.i', 'H.i',
    'DD.i.e', 'UD.i.e', 'H.e', 'DD.e.five', 'UD.e.five', 'H.five', 'DD.five.Shift.r',
    'UD.five.Shift.r', 'H.Shift.r', 'DD.Shift.r.o', 'UD.Shift.r.o', 'H.o', 'DD.o.a',
    'UD.o.a', 'H.a', 'DD.a.n', 'UD.a.n', 'H.n', 'DD.n.l', 'UD.n.l', 'H.l',
    'DD.l.Return', 'UD.l.Return', 'H.Return'
]

# Tworzenie wykresu
plt.figure(figsize=(12, 6))
for column in columns_to_plot:
    plt.plot(data_mean['index'], data_mean[column], label=column, alpha=0.7)

# Ustawienia wykresu
plt.title("Przewidywane wartości w funkcji indeksu")
plt.xlabel("Indeks (kolejność wpisania danych)")
plt.ylabel("Przewidywana wartość")
plt.legend(loc='upper right', fontsize='small', ncol=2)  # Legenda z wieloma liniami
plt.grid(True)
plt.show()
