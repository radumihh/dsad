# Rezolvări Subiecte Examen

Acest folder conține rezolvările pentru cele 4 subiecte de examen specificate.

## Structură

- `subiect_15.py`: Soluția pentru Subiectul 15 (Global Indicators, PCA)
- `subiect_35.py`: Soluția pentru Subiectul 35 (Air Quality, Clustering Variabile)
- `subiect_25.py`: Soluția pentru Subiectul 25 (Buget, PCA)
- `subiect_65.py`: Soluția pentru Subiectul 65 (Miscare Naturala, Clustering)

## Instrucțiuni de rulare

1. Asigurați-vă că aveți un folder numit `dataIN` în rădăcina proiectului sau lângă aceste scripturi (scripturile caută în `dataIN/`).
2. Copiați următoarele fișiere CSV în folderul `dataIN`:
   - `GlobalIndicatorsPerCapita.csv`
   - `ContriesIDs.csv` (sau `CountriesIDs.csv` daca e corectat)
   - `AirQualityCountries.csv`
   - `CountryCodes.csv`
   - `Buget.csv`
   - `LocPopulation.csv`
   - `DataSet_25.csv`
   - `NatLocMovement.csv`
   - `PopulationLoc.csv`
   - `DataSet_65.csv`
3. Rulați fiecare script:
   ```bash
   python subiect_15.py
   python subiect_35.py
   python subiect_25.py
   python subiect_65.py
   ```
4. Rezultatele vor fi salvate în folderul `dataOUT`.

## Note
- Scripturile folosesc metodele de seminar (înlocuire valori lipsă cu medie/mod, standardizare manuală, utilizare biblioteci standard).
- Codul gestionează erori de fișiere lipsă.
