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
mag = "all"
PBC = "false"

# ZAKRES OSI X
t_min = 0
t_max = 30

# ZAKRES OSI Y
y_min = -0.1
y_max = 0.5

fig, ax = plt.subplots(figsize=(10, 6))

energie_init = []

for i, Delta in enumerate(Delta_values):

    kolor = f"C{i % 10}"

    # WCZYTANIE DANYCH
    filename = f"Energia_L{L}_Delta{Delta}_J{J}_PBC{PBC}.data"
    dane = np.loadtxt(FOLDER / filename)

    t = dane[:, 0]
    epsilon = dane[:, 1]

    epsilon_init = epsilon[0]
    energie_init.append(epsilon_init)

    # WYKRES ENERGII
    ax.plot(
        t * J, epsilon,
        color=kolor,
        linewidth=2,
        label=fr"$\Delta={Delta}$"
    )

    # ENERGIA POCZĄTKOWA
    ax.axhline(
        epsilon_init,
        color=kolor,
        linestyle="--",
        linewidth=1,
        alpha=0.6
    )

# ZAKRESY OSI
ax.set_xlim(t_min, t_max)
ax.set_ylim(y_min, y_max)

# DODANIE ENERGII POCZĄTKOWYCH DO OSI Y
yticks = np.unique(np.round(
    np.concatenate([ax.get_yticks(), energie_init]), 6
))

yticks = yticks[(yticks >= y_min) & (yticks <= y_max)]

ax.set_yticks(yticks)
ax.set_yticklabels([f"{y:g}" for y in yticks])

# POGRUBIENIE WARTOŚCI ENERGII POCZĄTKOWYCH
for tick, wartosc in zip(ax.get_yticklabels(), yticks):

    for i, epsilon_init in enumerate(energie_init):

        if np.isclose(wartosc, epsilon_init, atol=1e-6):
            tick.set_fontweight("bold")
            tick.set_color(f"C{i % 10}")

# PODPISY OSI
ax.set_xlabel(r"$t$", fontsize=13)
ax.set_ylabel(r"$\langle E(t)\rangle/L$", fontsize=13)

ax.tick_params(axis="both", labelsize=12)

# TYTUŁ
ax.set_title(
    rf"XXZ: $L={L}$, $J={J}$, mag={mag}, PBC={PBC}",
    fontsize=13
)

ax.grid(alpha=0.3)
ax.legend(fontsize=13)

plt.tight_layout()

# ZAPIS WYKRESU
FOLDER_WYKRESY = FOLDER.parent / "plots"
FOLDER_WYKRESY.mkdir(exist_ok=True)

Delta_str = "-".join(map(str, Delta_values))

plt.savefig(
    FOLDER_WYKRESY / f"Energia_vs_t_L{L}_Delta{Delta_str}_J{J}_PBC{PBC}.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()