from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# USTAWIENIA
# =============================================================================

FOLDER_DANE = Path(
    r"C:\Users\aleks\Desktop\PWR\Numeryczne metody badania układów kwantowych\Pro_3_XXZ_ewolucja_czasowa\data"
)

# Liczba krokow czasowych - wpisywana recznie
nt = 201

# Skala kolorow
vmin = -0.5
vmax = 0.5

cmap = "RdBu_r"

# Zapis wykresow
SAVE_PLOTS = True

# =============================================================================
# ODCZYT PARAMETROW Z NAZWY PLIKU
# =============================================================================

def extract_parameters(filename):

    pattern = (
        r"Sz_sciana_"
        r"L(?P<L>\d+)_"
        r"Delta(?P<Delta>[^_]+)_"
        r"J(?P<J>[^_]+)_"
        r"PBC(?P<PBC>true|false)"
    )

    match = re.search(pattern, filename.stem, re.IGNORECASE)

    if match is None:
        raise ValueError(
            f"Nie mozna odczytac parametrow z nazwy pliku: {filename.name}"
        )

    return match.groupdict()


# =============================================================================
# RYSOWANIE WYKRESU
# =============================================================================

def plot_evolution(filename):

    # Odczyt parametrow
    p = extract_parameters(filename)

    L = int(p["L"])
    Delta = p["Delta"]
    J = p["J"]
    PBC = p["PBC"].lower()

    # Wczytanie danych
    data = np.loadtxt(filename, ndmin=2)

    if data.shape[1] != L + 1:
        raise ValueError(
            f"Nieprawidlowy rozmiar danych w {filename.name}. "
            f"Oczekiwano {L + 1} kolumn, otrzymano {data.shape[1]}."
        )

    t = data[:, 0]
    Sz = data[:, 1:]

    # Krawedzie komorek na osi czasu
    if len(t) > 1:
        dt = np.diff(t)

        if np.any(dt <= 0):
            raise ValueError("Czas musi byc rosnacy.")

        t_edges = np.empty(len(t) + 1)
        t_edges[1:-1] = (t[:-1] + t[1:]) / 2
        t_edges[0] = t[0] - dt[0] / 2
        t_edges[-1] = t[-1] + dt[-1] / 2

    else:
        t_edges = np.array([t[0] - 0.5, t[0] + 0.5])

    x_edges = np.arange(0.5, L + 1.5, 1)

    # =========================================================================
    # WYKRES
    # =========================================================================

    fig, ax = plt.subplots(figsize=(10, 6))

    pcm = ax.pcolormesh(
        x_edges,
        t_edges,
        Sz,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat"
    )

    # Colorbar
    cbar = fig.colorbar(
        pcm,
        ax=ax,
        ticks=np.linspace(vmin, vmax, 5)
    )

    cbar.set_label(
        r"$\langle S_i^z(t)\rangle$",
        fontsize=13
    )

    # Osie
    ax.set_xlabel("Numer wezla", fontsize=13)
    ax.set_ylabel("Czas (j.u.)", fontsize=13)

    ax.set_xticks(np.arange(1, L + 1))

    # Tytul
    ax.set_title(
        rf"$L={L},\ \Delta={Delta},\ J={J},\ "
        rf"\mathrm{{PBC}}={PBC},\ n_t={nt}$",
        fontsize=14,
        pad=12
    )

    ax.tick_params(axis="both", labelsize=11)

    fig.tight_layout()

    # Zapis
    if SAVE_PLOTS:
        output_path = filename.with_suffix(".png")
        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )
        print(f"Zapisano: {output_path}")

    print(
        f"Plik: {filename.name}\n"
        f"L = {L}, Delta = {Delta}, J = {J}, PBC = {PBC}\n"
        f"Liczba punktow czasowych w pliku = {len(t)}\n"
    )

    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():

    files = sorted(FOLDER_DANE.glob("Sz_sciana_*.data"))

    if not files:
        raise FileNotFoundError(
            f"Nie znaleziono plikow Sz_sciana_*.data w:\n{FOLDER_DANE}"
        )

    print(f"Znaleziono {len(files)} plikow.\n")

    for filename in files:
        plot_evolution(filename)

    plt.show()


if __name__ == "__main__":
    main()
