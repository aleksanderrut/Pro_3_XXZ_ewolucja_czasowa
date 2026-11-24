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

# Sciana domenowa - stan początkowy
function gen_stan_sciany(L::Int, baza_int, indeks_odwrotny)
    state_sciana = (1 << (L ÷ 2)) - 1
    indeks_sciany = indeks_odwrotny[state_sciana]

    # Stan początkowy w bazie Hamiltonianu
    psi = zeros(ComplexF64, length(baza_int))
    psi[indeks_sciany] = 1.0 + 0im
    return psi
end

# Operatory lokalne Sz w bazie Hamiltonianu
function gen_Sz_diag(L::Int, baza_int)
    ile_stanow = length(baza_int)
    # Wiersze = pozycje spinów; Kolumny = stany bazowe
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

# Dobieranie beta na podstawie energii początkowej
function dobierz_beta(energie, epsilon_init, L)
    beta_left = -100.0
    beta_right = 100.0
    tol = 1e-6
    # Energia kanoniczna dla zadanego beta
    function energia_kanoniczna(beta)
        x = -beta .* energie
        wagi = exp.(x .- maximum(x))
        return sum(energie .* wagi) / (L * sum(wagi))
    end
    # Metoda bisekcji - rozwiązanie rówania
    beta = (beta_left + beta_right) / 2
    epsilon = energia_kanoniczna(beta) #definicja funkcji = 0
    while abs(epsilon - epsilon_init) > tol
        if epsilon > epsilon_init
            beta_left = beta
        else
            beta_right = beta
        end
        beta = (beta_left + beta_right) / 2
        epsilon = energia_kanoniczna(beta)
    end
    return beta
end

# Zapis do pliku .data
function save_data(data, filename)
    folder = raw"C:\Users\aleks\Desktop\PWR\Numeryczne metody badania układów kwantowych\Pro_3_XXZ_ewolucja_czasowa\data"
    filename *= ".data"
    filepath = joinpath(folder, filename)
    writedlm(filepath, data)
    println("Zapisano do pliku:")
    println(filepath)
end

function main()
    # PARAMETRY MODELU
    J = 1.0
    # mag = :all
    mag = 0
    PBC = true
    Delta_values = [0.0, 1.0, 2.0]
    #Delta_values = [0.0]
    L_values = [14]
    #L_values = [14,15,16]
    # PARAMETRY CZASU
    t_min = 0.0
    t_max = 30.0
    nt = 301
    t_values = collect(range(t_min, t_max, length=nt))

    # MACIERZ NA WYNIKI SZUKANIA TEMPERATUR
    wyniki_T = zeros(Float64, length(Delta_values) * length(L_values), 5)
    indeks_T = 1
    # PĘTLA po prametrach
    for Delta in Delta_values
        for L in L_values
            println("L = $L, Delta = $Delta")
            println()

            baza_int, baza_bit, indeks_odwrotny = gen_baza_xxz(L; mag=mag)
            H = gen_ham_xxz(L, J, Delta, PBC, baza_int, indeks_odwrotny)
            psi0 = gen_stan_sciany(L, baza_int, indeks_odwrotny)
            Sz_diag = gen_Sz_diag(L, baza_int)

            println("Diagonalizacja...")
            diago = eigen(Symmetric(H))
            energie = diago.values
            wektory = diago.vectors
            wsp = wektory' * psi0 # stan początkowy zapisany w bazie energii własnych

            # MACIERZE NA WYNIKI
            wyniki = zeros(Float64, nt, L + 1)
            wyniki_energia = zeros(Float64, nt, 2)
            # ENERGIA POCZĄTKOWA I WYZNACZENIE TEMPERATURY
            epsilon_init = sum(abs2.(wsp) .* energie) / L
            beta = dobierz_beta(energie, epsilon_init, L) # urzycie bisekcji abu znaleśc bete
            T = abs(beta) < 1e-12 ? Inf : 1 / beta
            println("Energia początkowa na spin = ", epsilon_init)
            println("Beta = ", beta)
            println("Temperatura T = ", T)

            # ZAPIS punktu temperatury i podobnych do pliku
            wyniki_T[indeks_T, 1] = T
            wyniki_T[indeks_T, 2] = epsilon_init
            wyniki_T[indeks_T, 3] = L
            wyniki_T[indeks_T, 4] = Delta
            wyniki_T[indeks_T, 5] = beta
            indeks_T += 1

            # właściwa ewolucja czasowa
            for (k, t) in enumerate(t_values)
                wsp_t = wsp .* exp.(-1im .* energie .* t) #e do iE_nt zeby talko raz to liczyć
                
                # Ewolucja ściany domenowej
                psi_t = wektory * wsp_t
                prawdopodobienstwa = abs2.(psi_t) # jakie stany są w aktualnym stanie układu Sz_diag - policzone wcześniej
                Sz_t = Sz_diag * prawdopodobienstwa # Lokalne wartości oczekiwane Sz 
                wyniki[k, 1] = t
                wyniki[k, 2:end] = Sz_t

                # Średnia energia na spin w chwili t
                epsilon_t = sum(abs2.(wsp_t) .* energie) / L
                wyniki_energia[k, 1] = t
                wyniki_energia[k, 2] = epsilon_t
            end

            # ZAPIS WYNIKÓW EWOLUCJI
            filename = "Sz_sciana_L$(L)_Delta$(Delta)_J$(J)_PBC$(PBC)"
            save_data(wyniki, filename)
            filename_energia = "Energia_L$(L)_Delta$(Delta)_J$(J)_PBC$(PBC)"
            save_data(wyniki_energia, filename_energia)

            #Tutaj cześć z funkcjami średniej wartość energii od temperatury
            T_min = 0.01
            T_max = 3.0
            n_plot = 601
            T_values_neg = range(-T_max, -T_min, length=n_plot)
            T_values_pos = range(T_min, T_max, length=n_plot)
            T_values = vcat(T_values_neg, T_values_pos) #jeden wspołny przedział

            wyniki_E_T = zeros(Float64, length(T_values), 2)
            for (k, T_plot) in enumerate(T_values)
                x = -(1/T_plot) .* energie
                wagi = exp.(x .- maximum(x))
                Z = sum(wagi)
                epsilon = sum(energie .* wagi) / (L * Z)

                wyniki_E_T[k, 1] = T_plot
                wyniki_E_T[k, 2] = epsilon
                filename_E_T = "Energia_vs_T_L$(L)_Delta$(Delta)_J$(J)_mag$(mag)_PBC$(PBC)"
                save_data(wyniki_E_T, filename_E_T)
            end
        end
    end
    # ZAPIS T zanlezionych na początku pętli
    filename_T = "Energia_init_vs_T_J$(J)_mag$(mag)_PBC$(PBC)"
    save_data(wyniki_T, filename_T)
end

main()
