from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

FOLDER = Path(
    r"C:\Users\aleks\Desktop\PWR\Numeryczne metody badania układów kwantowych\Pro_3_XXZ_ewolucja_czasowa\data"
)

# PARAMETRY
L = 14
Delta_values = [0.0, 1.0, 2.0]
J = 1.0
mag = 0
PBC = "true"

# ZAKRES OSI X
T_min = 0
T_max = 3

# WCZYTANIE PUNKTÓW POCZĄTKOWYCH
filename_init = f"Energia_init_vs_T_J{J}_mag{mag}_PBC{PBC}.data"
dane_init = np.loadtxt(FOLDER / filename_init, ndmin=2)

# TWORZENIE WYKRESÓW
fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for i, Delta in enumerate(Delta_values):
    kolor = f"C{i % 10}"

    # WCZYTANIE KRZYWEJ ENERGII
    filename = f"Energia_vs_T_L{L}_Delta{Delta}_J{J}_mag{mag}_PBC{PBC}.data"
    dane = np.loadtxt(FOLDER / filename)
    T = dane[:, 0]
    epsilon = dane[:, 1]

    # PODZIAŁ NA DODATNIE I UJEMNE TEMPERATURY
    pos = T > 0
    neg = T < 0

    # Dodatnie temperatury
    ax[0].plot(T[pos], epsilon[pos], color=kolor,
               linewidth=2, label=fr"$\Delta={Delta}$")

    # Ujemne temperatury
    ax[1].plot(-T[neg], epsilon[neg], color=kolor,
               linewidth=2, label=fr"$\Delta={Delta}$")

    # NANIESIENIE PUNKTU POCZĄTKOWEGO
    # Dla Delta = 0 pomijamy szukanie punktu
    if Delta != 0:
        punkt = dane_init[(dane_init[:, 2] == L) & (dane_init[:, 3] == Delta)][0]
        T_init = punkt[0]
        epsilon_init = punkt[1]

        if np.isfinite(T_init):
            if T_init > 0:
                ax[0].scatter(
                    T_init, epsilon_init, color=kolor,
                    edgecolor="black", s=70, zorder=5,
                    label=fr"$({T_init/J:.4f},\,{epsilon_init:.4f})$"
                )
            elif T_init < 0:
                ax[1].scatter(
                    -T_init, epsilon_init, color=kolor,
                    edgecolor="black", s=70, zorder=5,
                    label=fr"$({-T_init/J:.4f},\,{epsilon_init:.4f})$"
                )

# PODPISY OSI
ax[0].set_xlabel(r"$T$")
ax[1].set_xlabel(r"$-T$")
ax[0].set_ylabel(r"$\langle E\rangle/L$")
ax[1].set_ylabel(r"$\langle E\rangle/L$")
ax[1].tick_params(axis="y", labelleft=True)

# NAGŁÓWKI WYKRESÓW
ax[0].set_title(
    r"$\langle E\rangle=\frac{1}{L}\frac{1}{Z}"
    r"\sum_i \epsilon_i e^{-\beta\epsilon_i}$",
    fontsize=13, pad=15
)
ax[1].set_title(
    r"$\langle E\rangle=\frac{1}{L}\frac{1}{Z}"
    r"\sum_i \epsilon_i e^{+\beta\epsilon_i}$",
    fontsize=13, pad=15
)

# USTAWIENIA WYKRESÓW
for a in ax:
    a.set_xlim(T_min, T_max)
    a.grid(alpha=0.3)
    a.legend()

# TYTUŁ WSPÓLNY
fig.suptitle(
    rf"XXZ: $L={L}$, $J={J}$, mag={mag}, PBC={PBC}",
    fontsize=13
)
plt.tight_layout()

# ZAPIS WYKRESU
FOLDER_WYKRESY = FOLDER.parent / "plots"
FOLDER_WYKRESY.mkdir(exist_ok=True)
Delta_str = "-".join(map(str, Delta_values))

plt.savefig(
    FOLDER_WYKRESY / f"Energia_vs_T_L{L}_Delta{Delta_str}_J{J}_mag{mag}_PBC{PBC}.png",
    dpi=300, bbox_inches="tight"
)
plt.show()