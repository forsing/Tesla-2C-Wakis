"""
SRBIN Nikola Tesla, za sva vremena, najveci naucnik sveta.

SERBIAN Nikola Tesla, for all time, the greatest scientist in the world.
"""



"""
Tesla_Wakis_2C.py  —  GRUPA 2, varijanta 2C (Wakis 3D EM FIT motor)

Ista struktura kao GRUPA 1, 2A i 2B:
  motor (3D EM FIT longitudinalno polje)  ->  primena na 4630 izvlacenja  ->  skor  ->  rangirane kombinacije

Wakis je 3D elektromagnetni solver. Ovde koristim mali vakuum domen i
ubrizgavam uzduzni impuls u komponentu E_z. Zatim citam centralnu liniju
polja E_z u pravcu prostiranja:
  S(z)   = longitudinalna komponenta E_z duz centralne linije
  E_x    = -dS/dz  (u kodu zadrzavam ime E_x zbog zajednickog pipeline-a)
"""


import numpy as np

from Tesla_Scalar_1 import (
    SEED,
    W_TALAS,
    W_FREQ,
    CSV_PATH,
    OUTPUT_DIR,
    ucitaj_izvlacenja,
    glavne_mere,
    ne_frekvencijski_skor,
    frekvencija_brojeva,
    kombinovani_skor,
    izaberi_kombinacije,
    skor_kombinacije,
    nacrtaj_polje,
)

OSNOVA = "tesla_Wakis_2C"


def _normalizuj_signal(S):
    """Centriraj i skaliraj signal na stabilan opseg."""
    S = np.asarray(S, dtype=float)
    S = S - np.mean(S)
    m = np.max(np.abs(S))
    if m <= 0:
        return S
    return S / m


def simuliraj_wakis(nx=4630, Nx=18, Ny=18, Nz=128, Nt=220):
    """Wakis 3D FIT motor. Vrati (z, S, E_x) duz pravca prostiranja.

    U malom vakuum domenu ubrizgavam Gaussov impuls u longitudinalnu
    komponentu E_z blizu pocetka z-ose i pustam 3D EM solver da ga propagira.
    """
    from wakis import GridFIT3D, SolverFIT3D

    grid = GridFIT3D(
        -1.0, 1.0,
        -1.0, 1.0,
        0.0, 8.0,
        Nx, Ny, Nz,
        verbose=0,
    )
    solver = SolverFIT3D(
        grid=grid,
        cfln=0.5,
        bc_low=["pec", "pec", "abc"],
        bc_high=["pec", "pec", "abc"],
        use_stl=False,
        bg=[1.0, 1.0],
        verbose=0,
    )

    cx, cy = solver.Nx // 2, solver.Ny // 2
    z0 = max(2, solver.Nz // 8)
    sirina_z = solver.Nz * 0.035
    sirina_t = Nt * 0.09
    centar_t = Nt * 0.22
    z_idx = np.arange(solver.Nz)
    prostorni_profil = np.exp(-((z_idx - z0) ** 2) / (2.0 * sirina_z ** 2))

    for n in range(Nt):
        vremenski_profil = np.exp(-((n - centar_t) ** 2) / (2.0 * sirina_t ** 2))
        solver.E[cx, cy, :, "z"] += prostorni_profil * vremenski_profil
        solver.one_step()

    S_raw = np.asarray(solver.E[cx, cy, :, "z"])
    S_raw = _normalizuj_signal(S_raw)

    z = np.linspace(0.0, 1.0, nx)
    S = np.interp(z, np.linspace(0.0, 1.0, solver.Nz), S_raw)
    E_x = -np.gradient(S, z[1] - z[0])
    return z, S, E_x


def main():
    # --- Korak 1: motor (Wakis 3D EM FIT) ---
    izvlacenja = ucitaj_izvlacenja()
    n = len(izvlacenja)
    x, S, E_x = simuliraj_wakis(nx=n)
    mere = glavne_mere(S, E_x)
    print()
    print("Tesla Scalar / GRUPA 2 - 2C (Wakis 3D EM FIT motor)")
    print("Talas: longitudinalna komponenta E_z kroz 3D EM domen")
    print("Uzduzno izvedeno polje: E_x = -dS/dz (ime zadrzano zbog pipeline-a)")
    print()
    print(f"broj tacaka: {len(x)}")
    print(f"max S: {mere['max_S']:.10f}")
    print(f"max |E_x|: {mere['max_abs_E_x']:.10f}")
    print(f"ukupna gustina energije: {mere['ukupna_gustina_energije']:.10f}")
    print()

    # --- Korak 2: primena talasa na CSV + prava frekvencija ---
    energija = 0.5 * (S ** 2 + E_x ** 2)
    talas_skor, _ = ne_frekvencijski_skor(izvlacenja, energija)
    udeo, pojave = frekvencija_brojeva(izvlacenja)
    skor = kombinovani_skor(talas_skor, udeo)
    poredak = sorted(skor.items(), key=lambda kv: kv[1], reverse=True)
    freq_poredak = sorted(pojave, key=lambda b: (pojave[b], b), reverse=True)
    kombinacije = izaberi_kombinacije(skor, broj_kombinacija=10, seed=SEED)
    rangirane_kombinacije = sorted(
        ((k, skor_kombinacije(k, skor)) for k in kombinacije),
        key=lambda kv: kv[1],
        reverse=True,
    )
    png, jpg = nacrtaj_polje(x, S, E_x, osnova=OSNOVA)

    with open(OUTPUT_DIR / f"{OSNOVA}.txt", "w", encoding="utf-8") as f:
        f.write("Tesla Scalar - GRUPA 2 / 2C (Wakis 3D EM FIT: talas + prava frekvencija)\n")
        f.write(f"CSV: {CSV_PATH}\n")
        f.write(f"Izvlacenja: {n} | Seed: {SEED} | tezine: talas={W_TALAS} freq={W_FREQ}\n\n")
        f.write("Brojevi po kombinovanom skoru (tezinski talas + frekvencija):\n")
        for b, s in poredak:
            f.write(f"  {b:02d}  skor={s:.10f}  freq={udeo[b]:.5f}  (pojava={pojave[b]})\n")

        f.write("\nTabela pravih frekvencija (opadajuce po freq, pa po broju):\n")
        f.write("  broj | pojava |   udeo\n")
        f.write("  -----+--------+--------\n")
        for b in freq_poredak:
            f.write(f"   {b:02d}  |  {pojave[b]:4d}  | {udeo[b]:.5f}\n")
        f.write(f"  ukupno pojava: {sum(pojave.values())}\n")

        f.write("\nPredlozene kombinacije (rangirane po skoru kombinacije):\n")
        for i, (k, s_komb) in enumerate(rangirane_kombinacije, start=1):
            f.write(f"  {i:02d}. " + " ".join(f"{v:02d}" for v in k) + f"  skor_komb={s_komb:.10f}\n")

        f.write("\nSlike talasa/polja:\n")
        f.write(f"  PNG: {png}\n")
        f.write(f"  JPG: {jpg}\n")

    print()
    print("\nTesla Scalar - GRUPA 2 / 2C (Wakis 3D EM FIT: talas + prava frekvencija)")
    print(f"CSV: {CSV_PATH} | Izvlacenja: {n} | tezine: talas={W_TALAS} freq={W_FREQ}")
    print("\nTop 10 brojeva po kombinovanom skoru (tezinski talas + frekvencija):")
    for b, s in poredak[:10]:
        print(f"  {b:02d}  skor={s:.10f}  freq={udeo[b]:.5f}  (pojava={pojave[b]})")

    print()
    print("\nTabela pravih frekvencija (opadajuce po freq, pa po broju):")
    print("  broj | pojava |   udeo")
    print("  -----+--------+--------")
    for b in freq_poredak:
        print(f"   {b:02d}  |  {pojave[b]:4d}  | {udeo[b]:.5f}")
    print(f"  ukupno pojava: {sum(pojave.values())}")

    print()
    print("\nPredlozene kombinacije (rangirane po skoru kombinacije):")
    for i, (k, s_komb) in enumerate(rangirane_kombinacije, start=1):
        print(f"  {i:02d}. " + " ".join(f"{v:02d}" for v in k) + f"  skor_komb={s_komb:.10f}")
    print(f"\nSacuvano: {OUTPUT_DIR / f'{OSNOVA}.txt'}")
    print()


if __name__ == "__main__":
    main()



"""
Tesla Scalar / GRUPA 2 - 2C (Wakis 3D EM FIT motor)
Talas: longitudinalna komponenta E_z kroz 3D EM domen
Uzduzno izvedeno polje: E_x = -dS/dz (ime zadrzano zbog pipeline-a)

broj tacaka: 4630
max S: 0.9997040565
max |E_x|: 34.7740579998
ukupna gustina energije: 182502.6894896694

Slika talasa: /Tesla/tesla_Wakis_2C.png
Slika talasa: /Tesla/tesla_Wakis_2C.jpg


Tesla Scalar - GRUPA 2 / 2C (Wakis 3D EM FIT: talas + prava frekvencija)
CSV: /data/loto7hh_4630_k46.csv | Izvlacenja: 4630 | tezine: talas=0.7 freq=0.3

Top 10 brojeva po kombinovanom skoru (tezinski talas + frekvencija):
  24  skor=0.8541666667  freq=0.02592  (pojava=840)
   x  skor=0.7354933135  freq=0.02792  (pojava=905)
  35  skor=0.6452173395  freq=0.02601  (pojava=843)
   y  skor=0.6408397386  freq=0.02549  (pojava=826)
  28  skor=0.6238711571  freq=0.02530  (pojava=820)
   z  skor=0.6080282434  freq=0.02620  (pojava=849)
  38  skor=0.5995116427  freq=0.02598  (pojava=842)
   x  skor=0.5991339730  freq=0.02808  (pojava=910)
  22  skor=0.5527000557  freq=0.02626  (pojava=851)
  16  skor=0.5415414573  freq=0.02583  (pojava=837)


Tabela pravih frekvencija (opadajuce po freq, pa po broju):
  broj | pojava |   udeo
  -----+--------+--------
   08  |   910  | 0.02808
    x  |   905  | 0.02792
   34  |   873  | 0.02694
    y  |   869  | 0.02681
   37  |   860  | 0.02654
    z  |   860  | 0.02654
   32  |   857  | 0.02644
    x  |   854  | 0.02635
   22  |   851  | 0.02626
    y  |   849  | 0.02620
   29  |   848  | 0.02616
    z  |   845  | 0.02607
   35  |   843  | 0.02601
    x  |   843  | 0.02601
   38  |   842  | 0.02598
    y  |   842  | 0.02598
   24  |   840  | 0.02592
    z  |   839  | 0.02589
   16  |   837  | 0.02583
   31  |   830  | 0.02561
   13  |   828  | 0.02555
   05  |   828  | 0.02555
   21  |   826  | 0.02549
   03  |   825  | 0.02546
   02  |   824  | 0.02542
   28  |   820  | 0.02530
   18  |   820  | 0.02530
   06  |   816  | 0.02518
   19  |   813  | 0.02508
   04  |   812  | 0.02505
   12  |   810  | 0.02499
   14  |   809  | 0.02496
   15  |   797  | 0.02459
   27  |   788  | 0.02431
   01  |   788  | 0.02431
   30  |   787  | 0.02428
   36  |   786  | 0.02425
   20  |   770  | 0.02376
   17  |   766  | 0.02363
  ukupno pojava: 32410


Predlozene kombinacije (rangirane po skoru kombinacije):
  01. 07 x 13 y 23 z 32  skor_komb=4.0410091171
  02. 09 x 21 y 26 z 38  skor_komb=3.9749282633
  03. 16 x 25 y 28 z 33  skor_komb=3.9568111315
  04. 02 x 16 y 23 z 39  skor_komb=3.8853829246
  05. 08 x 22 y 31 z 33  skor_komb=3.8075387152
  06. 06 x 12 y 32 z 39  skor_komb=3.5336220850
  07. 05 x 23 y 29 z 36  skor_komb=3.4578392074
  08. 09 x 15 y 22 z 34  skor_komb=3.4182433035
  09. 05 x 13 y 29 z 39  skor_komb=3.3863421569
  10. 01 x 07 y 14 z 39  skor_komb=2.7786932349

Sacuvano: /Tesla/tesla_Wakis_2C.txt
"""



"""
Postoji teorijska osnova (EED/SLW) + alati za simulaciju talasa. 


GRUPA 2 

2. Gotove biblioteke za simulaciju talasa/EM polja (na njima bi se gradilo)


2A k-wave-python — simulacija talasnih polja (akustika, FDTD)

2B pycharge — EM polja/potencijali pokretnih naboja (JAX, GPU)

2C Wakis — 3D EM solver (računa i longitudinalne komponente)
           (najbliže Teslinom SLW)

2D rfx — diferencijabilni 3D FDTD EM simulator
         (može učenje/optimizacija)

k-wave-python — prvo. Najbliže grupi 1 (FDTD talasno polje).
pycharge — drugo. Uvodi prava polja naboja (JAX/GPU), dobra provera da li „izvor" menja rezultat.
Wakis — treće. Pravi 3D EM solver sa longitudinalnim komponentama → ovo je srce Tesline SLW priče.
rfx — poslednje. Diferencijabilni FDTD → kad sve radi, njime optimizujem parametre (učenje težina, ne ručno 0.7/0.3).

Logika: 
prve dve daju temelj i poređenje, treća donosi pravi longitudinalni talas, četvrta pretvara ceo sistem u nešto što se može podešavati/učiti.

Svaka varijanta = ista struktura kao grupa 1 (motor → primena na 4630 → skor → rangirane kombinacije), samo jači motor.
"""



"""
pravi 3D EM solver 
— daje najoštrije polje od svih: max|E_x| ≈ 34.8, energija ≈ 182.503 (daleko najjači gradijent).

Novi favorit broj: 24 (#1, skor 0.854) iako mu je frekvencija srednja (840) → čist talasni efekat.
23 je #2 i tu se talas i frekvencija slažu (frekvencijski #2).
28, x, 21 su podignuti talasom (frekvencijski ispod proseka).
34, koji je dominirao u 1 i 2A, ovde nije u top 10 → Wakis daje zaista nezavisan signal.
Favorit 2C: 07 x 13 y 23 z 32 (skor_komb = 4.0410).

Imam sad 4 nezavisna motora (1, 2A, 2B, 2C) — svaki daje svoj potpis, a struktura je svuda ista. 
"""



"""
Analiza — Tesla 2C (Wakis 3D EM FIT motor)

Motor: Wakis 3D elektromagnetni solver, FIT metoda. 
Ovo je do sada najbliže onome šta sam tražio za Teslinu longitudinalnu priču: 
pravi 3D EM domen i longitudinalna komponenta E_z. 
Iz centralne linije E_z dobijam S, pa izvod -dS/dz.

Mere polja: max S ≈ 0.9997, max |E_x| ≈ 34.77, energija ≈ 182502.69.

To je najjači gradijent do sada:

Tesla 1: blag 1D talas
2A: FDTD talasni solver
2B: EM potencijal naboja
2C: pravi 3D EM longitudinalni signal

Top brojevi: 24 · x · 35 · y · 28 · z · 38 · x · 22 · 16

24 je #1, a frekvencijski je tek sredina (840) → jak talasni signal.
23 je #2 i frekvencijski je takođe jak (905) → tu se talas i frekvencija poklapaju.
28, x, 38 su podignuti talasom, ne frekvencijom.
34, dominantan u Tesla 1 i 2A, ovde nije top 10 → Wakis ne kopira prethodne modele.
08 ostaje prisutan, ali nije vodeći — frekvencija mu pomaže, ali talas ga ne stavlja na vrh.
Favorit kombinacija 2C: 07 x 13 y 23 z 32 skor_komb = 4.0410

Zaključak: 2C je najfizičkiji i najbliži Teslinom SLW smeru jer koristi 3D EM polje i longitudinalnu komponentu. 
Njegov rezultat treba posebno čuvati kao „teški model". 
Nije nužno najbolji za kombinacije sam po sebi, ali je najvažniji za teorijsku stranu projekta.
"""



"""
source ~/tesla_env/bin/activate

Bitne verzije za tesla_env:

Paket	Verzija
python  3.11.13
numpy   2.2.6
scipy   1.15.3
pandas  3.0.3
matplotlib    3.10.9
k-Wave-python 0.6.2
pycharge      2.0.1
jax        0.10.1
jaxlib     0.10.1
jaxtyping  0.3.7
equinox    0.13.8
lineax     0.1.1
optimistix 0.1.0
ml-dtypes
(uz jax)
opencv-python 4.13.0.92
h5py          3.16.0
"""
