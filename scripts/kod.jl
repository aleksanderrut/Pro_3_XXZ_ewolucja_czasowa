using SparseArrays
using LinearAlgebra
using DelimitedFiles
using SparseArrays

# Generowanie bazy XXZ
function gen_baza_xxz(L::Int; mag=:all)
    # Rozmiar bazy w zależności od trybu
    rozmiar_pelnej_bazy = 1 << L # przesówanie bitowe jedynki w lewo o L mijesc
    if mag == :all
        liczba_stanow = rozmiar_pelnej_bazy
        sektory = 0:L #sektor zdefiniowany przez liczbę N_up w nim
    elseif mag isa Real
        N_up = L/2 + mag # N_up = (N_up+N_down)/2 + (N_up - N_down)/2 
        N_up = Int(N_up) # tka dla pewność
        liczba_stanow = binomial(L, N_up)
        sektory = N_up:N_up #sektor zdefiniowany prz liczbę N_up w nim np.3:3 -> 3
    end
    # Tworzenie baz z uwzgledeniem sektorów magnetyzacji
    baza_int = zeros(Int, liczba_stanow)
    i = 1 # bo Julia
    for N_up_sektor in sektory
        for state in 0:(rozmiar_pelnej_bazy - 1)
            if count_ones(state) == N_up_sektor
                baza_int[i] = state
                i = i + 1
            end
        end
    end
    # Odwzorowanie odworotne stan -> indeks w bazie, zapisanw jako słownik
    indeks_odwrotny = Dict{Int, Int}() # typ słownik w juli
    for (i, state) in enumerate(baza_int)
        indeks_odwrotny[state] = i
    end
    println("LICZBA STANOW: ")
    println(liczba_stanow)
    # Zamiana bazy na zapis binarny
    baza_bit = [string(state, base=2, pad=L) for state in baza_int]
    return baza_int, baza_bit, indeks_odwrotny
end

# Generowanie ham XXZ
function gen_ham_xxz(L::Int, J::Real, Delta::Real, PBC::Bool, baza_int::Vector{Int}, indeks_odwrotny::Dict{Int, Int})
    #tworzenie macierzy Hamiltonianu
    ile_stanow = length(baza_int)
    H = zeros(Float64, ile_stanow, ile_stanow)

    # Liczba połączeń między sąsiednimi spinami
    liczba_polaczen = L - 1
    if PBC == true && L > 2
        liczba_polaczen += 1
    end

    #pętla po stanch
    for i in 1:length(baza_int)
        state = baza_int[i]
        #pętla po możliwych przeskokach
        for bit_s in 0:(liczba_polaczen - 1)
            #struktura poąłczeń (dla PBC ostatni łączy się z pierwszym)
            if bit_s == L - 1 
                bit_t = 0
            else
                bit_t = bit_s + 1
            end
            #odczyt spinów z uzyciem bitowego and (bit&1) 0 = down, 1 = up
            spin_s = (state >> bit_s) & 1
            spin_t = (state >> bit_t) & 1

            # CZĘŚĆ DIAGONALNA
            if spin_s == spin_t
                H[i, i] += (J*Delta)/4
            else
                H[i, i] -= (J*Delta)/4
            end
            # CZĘŚĆ POZADIAGONALNA
            if spin_s != spin_t
                # Zamiana dwóch spinów przez XOR - ⊻ a to | - to or
                new_state = state ⊻ ((1 << bit_s) | (1 << bit_t))
                new_index = indeks_odwrotny[new_state] # Indeks nowego stanu w bazie
                H[new_index, i] += J / 2
            end
        end
    end
    return H
end

# =============================================================================
# STAN POCZĄTKOWY - ŚCIANKA DOMENOWA
# =============================================================================

function gen_stan_sciany(L::Int, baza_int, indeks_odwrotny)

    iseven(L) || error("Sciana domenowa wymaga parzystego L")

    # Pierwsza połowa spinów UP, druga połowa DOWN.
    # Spin na pozycji 1 odpowiada najmniej znaczącemu bitowi.
    # Dla L=6: stan 000111, czyli fizycznie ↑↑↑↓↓↓.

    state_sciana = (1 << (L ÷ 2)) - 1

    haskey(indeks_odwrotny, state_sciana) ||
        error("Stan sciany nie nalezy do wybranego sektora")

    indeks_sciany = indeks_odwrotny[state_sciana]

    # Stan początkowy w bazie Hamiltonianu
    psi = zeros(ComplexF64, length(baza_int))
    psi[indeks_sciany] = 1.0 + 0im

    return psi
end

# =============================================================================
# OPERATORY LOKALNE Sz - ICH ELEMENTY DIAGONALNE
# =============================================================================

function gen_Sz_diag(L::Int, baza_int)

    ile_stanow = length(baza_int)

    # Wiersze = pozycje spinów
    # Kolumny = stany bazowe
    Sz_diag = zeros(Float64, L, ile_stanow)

    for (k, state) in enumerate(baza_int)

        for site in 1:L

            spin = (state >> (site - 1)) & 1

            if spin == 1
                Sz_diag[site, k] = 0.5
            else
                Sz_diag[site, k] = -0.5
            end
        end
    end

    return Sz_diag
end


# =============================================================================
# EWOLUCJA CZASOWA
# =============================================================================

function ewolucja_czasowa(H, psi0, Sz_diag, t_values)

    ile_stanow = length(psi0)
    L = size(Sz_diag, 1)
    nt = length(t_values)

    dt = t_values[2] - t_values[1]

    # Macierz jednostkowa
    Id = spdiagm(0 => ones(Float64, ile_stanow))

    # Metoda Cranka-Nicolsona:
    # (I + i*dt*H/2) psi(t+dt) = (I - i*dt*H/2) psi(t)

    A = Id + (1im * dt / 2) * H
    B = Id - (1im * dt / 2) * H

    # Faktoryzacja wykonywana tylko raz
    F = lu(A)

    psi = copy(psi0)

    # Wyniki: wiersze = czas, kolumny = pozycja spinu
    Sz_t = zeros(Float64, nt, L)
    Sz_total_t = zeros(Float64, nt)

    max_blad_normy = 0.0

    for it in 1:nt

        # Prawdopodobieństwa poszczególnych stanów bazowych
        prawdopodobienstwa = abs2.(psi)

        # <Sz_i(t)> dla wszystkich spinów
        Sz_t[it, :] .= Sz_diag * prawdopodobienstwa

        # Całkowita magnetyzacja
        Sz_total_t[it] = sum(Sz_t[it, :])

        # Kontrola normy funkcji falowej
        max_blad_normy = max(
            max_blad_normy,
            abs(norm(psi) - 1.0)
        )

        # Przejście do następnej chwili czasu
        if it < nt
            psi = F \ (B * psi)
        end
    end

    println("Maksymalny blad normy: ", max_blad_normy)
    println("Sz_total na koncu: ", Sz_total_t[end])

    return Sz_t, Sz_total_t
end


# =============================================================================
# MAIN
# =============================================================================

function main()

    J = 1.0

    # Ścianka domenowa ma całkowitą magnetyzację zero
    mag = 0

    PBC = false

    Delta_values = [0.5, 1.0, 2.0]
    L_values = [2, 4, 6, 8, 10, 12, 14]

    # PARAMETRY CZASU
    t_min = 0.0
    t_max = 20.0
    nt = 201

    t_values = collect(range(t_min, t_max, length=nt))

    # Pętla po długościach łańcucha
    for L in L_values

        baza_int, baza_bit, indeks_odwrotny =
            gen_baza_xxz(L; mag=mag)

        # Stan początkowy
        psi0 = gen_stan_sciany(L, baza_int, indeks_odwrotny)

        # Elementy diagonalne lokalnych operatorów Sz
        Sz_diag = gen_Sz_diag(L, baza_int)

        # Pętla po anizotropiach
        for Delta in Delta_values

            println("\n--------------------------------")
            println("L = ", L, " | Delta = ", Delta)
            println("--------------------------------")

            H = gen_ham_xxz(
                L, J, Delta, PBC,
                baza_int, indeks_odwrotny
            )

            # Ewolucja i wartości oczekiwane
            Sz_t, Sz_total_t = ewolucja_czasowa(
                H, psi0, Sz_diag, t_values
            )

            # Kolumny: t, Sz_1, Sz_2, ..., Sz_L, Sz_total
            wyniki = hcat(t_values, Sz_t, Sz_total_t)

            filename = "Sz_t_PBC$(PBC)_L$(L)_J$(J)_Delta$(Delta)_mag$(mag)_tmax$(t_max)_nt$(nt)"

            save_data(wyniki, filename)
        end
    end
end

# Zapis do pliku .data
function save_data(data, filename)
    folder = raw"C:\Users\aleks\Desktop\PWR\Numeryczne metody badania układów kwantowych\Pro_2_XXZ_pojemność_cieplna\data"
    filename *= ".data"
    filepath = joinpath(folder, filename)
    writedlm(filepath, data)
    println("Zapisano do pliku:")
    println(filepath)
end

function main()
    J = 1.0
    mag = :all
    # mag = 0
    PBC = false

    Delta_values = [0.5, 1.0, 2.0]
    # Delta_values = [1.0]
    #L_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    L_values = [2, 4, 6, 8, 10, 12, 14]
    # L_values = [2]

    # PARAMETRY CZASU
    t_min = 0.0
    t_max = 20.0
    nt = 201
    t_values = collect(range(t_min, t_max, length=nt))

    # Pętla po długościach łańcucha
    for L in L_values

        baza_int, baza_bit, indeks_odwrotny =
            gen_baza_xxz(L; mag=mag)

        # Stan początkowy
        psi0 = gen_stan_sciany(L, baza_int, indeks_odwrotny)

        # Elementy diagonalne lokalnych operatorów Sz
        Sz_diag = gen_Sz_diag(L, baza_int)

        # Pętla po anizotropiach
        for Delta in Delta_values

            println("\n--------------------------------")
            println("L = ", L, " | Delta = ", Delta)
            println("--------------------------------")

            H = gen_ham_xxz(
                L, J, Delta, PBC,
                baza_int, indeks_odwrotny
            )

            # Ewolucja i wartości oczekiwane
            Sz_t, Sz_total_t = ewolucja_czasowa(
                H, psi0, Sz_diag, t_values
            )

            # Kolumny: t, Sz_1, Sz_2, ..., Sz_L, Sz_total
            wyniki = hcat(t_values, Sz_t, Sz_total_t)

            filename = "Sz_t_PBC$(PBC)_L$(L)_J$(J)_Delta$(Delta)_mag$(mag)_tmax$(t_max)_nt$(nt)"

            save_data(wyniki, filename)
        end
    end
end

main()
