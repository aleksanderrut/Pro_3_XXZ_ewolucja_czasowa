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
    # mag = :all
    mag = 0
    PBC = false

    Delta_values = [0.5, 1.0, 2.0]
    # Delta_values = [1.0]
    #L_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    L_values = [2, 4, 6, 8, 10, 12, 14]
    # L_values = [2]

    for Delta in Delta_values
        #tabela na maksima, format: L, max_Cv, T_Cv_max
        wyniki_max_Cv = zeros(Float64, length(L_values), 3)

        for (idx_L, L_x) in enumerate(L_values)
            #Diagonalizacja
            println("Obliczenia: Delta = $Delta, L = $L_x")
            baza_int, baza_bit, indeks_odwrotny = gen_baza_xxz(L_x; mag)
            #save_data(baza_bit, "baza_PBC$(PBC)_L_x$(L_x)_Delta$(Delta)_J$(J)_mag$(mag)")
            H = gen_ham_xxz(L_x, J, Delta, PBC, baza_int, indeks_odwrotny)
            #save_data(H, "ham_PBC$(PBC)_L_x$(L_x)_Delta$(Delta)_J$(J)_mag$(mag)")
            eig = eigen(Symmetric(H))
            wartosci_wlasne = eig.values
            #wektory_wlasne = eig.vectors
            save_data(wartosci_wlasne, "eigenvalues_PBC$(PBC)_L_x$(L_x)_Delta$(Delta)_J$(J)_mag$(mag)")

            #Pojemność Cieplna Cv
            E = wartosci_wlasne
            E0 = wartosci_wlasne[1]
            ile_T = 200
            T_tablica = collect(range(0.05, 2.0, ile_T))
            wyniki_Cv = zeros(Float64, ile_T, 4)
            for i in 1:ile_T
                T = T_tablica[i]
                #suma statystyczna
                w = exp.(-(E .- E0) ./ T)
                Z = sum(w)
                #średnia energia <E>
                E_mean = sum(E .* w) / Z
                #średni kwadrat energii <E²>
                E2_mean = sum(E.^2 .* w) / Z
                #pojemność cieplna to już w pełni z wekotrami
                Cv = (E2_mean - E_mean^2) / T^2
                #tablica wyników:
                wyniki_Cv[i, 1] = T
                wyniki_Cv[i, 2] = E_mean
                wyniki_Cv[i, 3] = E2_mean
                wyniki_Cv[i, 4] = Cv
            end
            save_data(wyniki_Cv, "Cv_PBC$(PBC)_L_x$(L_x)_Delta$(Delta)_J$(J)_mag$(mag)")

            #znajdowanie maksium 
            max_Cv, idx_max = findmax(wyniki_Cv[:, 4])
            T_Cv_max = T_tablica[idx_max]
            #zapis do tabeli
            wyniki_max_Cv[idx_L, 1] = L_x
            wyniki_max_Cv[idx_L, 2] = max_Cv
            wyniki_max_Cv[idx_L, 3] = T_Cv_max
        end
        save_data(wyniki_max_Cv, "Cv_max_PBC$(PBC)_Delta$(Delta)_J$(J)_mag$(mag)")
    end
end

main()
