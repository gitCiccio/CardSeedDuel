# CardSeedDuel - Card Suit Game con DeepProbLog

Progetto di NeuroSymbolic AI 2026/2027 partecipanti: Cozza Simone, Foti Sofia, Gligora Ernesto, Tartaglia Francesco.

## Traccia e obiettivo

Date due immagini di carte, prevedere **Player 1 Wins**, **Player 2 Wins** oppure **Draw**.
Conta soltanto il seme: il valore della carta non influisce sul risultato.
La traccia originale e in [docs/CardSeedDuelAssignment_DeepProbLog.pdf](docs/CardSeedDuelAssignment_DeepProbLog.pdf).

Regole: **Cuori batte Fiori**, **Fiori batte Picche**, **Picche batte Cuori**; semi uguali danno pareggio.
Usare solo Hearts, Clubs e Spades; escludere Diamonds e Joker dal dataset usato negli esperimenti.

Realizzare e confrontare:

1. **Approccio A - DeepProbLog:** una rete PyTorch produce una distribuzione sui tre semi per ogni immagine; un predicato neurale e le regole ProbLog calcolano la distribuzione sui risultati. Il training puo propagare l'errore sul risultato fino alla rete percettiva.
2. **Approccio B - baseline neurale:** una rete riceve entrambe le immagini e predice direttamente il risultato della partita.

Costruire esempi `(I1, I2, outcome)` etichettati usando le regole. Valutare accuracy e F1 sul test e confrontare tempo di training ed efficienza computazionale. Ripetere gli esperimenti con **20%, 50% e 100% degli esempi di training**, mantenendo validation e test invariati. Specificare nel protocollo come si calcola F1 (consigliato macro-F1), seed, hardware e misure dei tempi.

Consegna prevista dalla traccia: codice sorgente e presentazione di **8-12 slide**, con teoria DeepProbLog, metodi, risultati e confronto, entro il **18 dicembre**; presentazione del gruppo nella stessa giornata.
La traccia usa l'espressione "ASP rules": nell'implementazione DeepProbLog i file logici useranno la sintassi ProbLog.

## Struttura del repository

```text
Neurosymbolic_project/
|-- README.md
|-- .gitignore
|-- requirements.txt                # installa il package e le dipendenze da pyproject.toml
|-- configs/                        # configurazioni riproducibili degli esperimenti
|-- data/                           # dati locali, separati dal codice
|   |-- raw/
|   |   |-- cards.csv               # metadati originali, esclusi da Git
|   |   |-- train/                  # immagini per apprendimento dei pesi
|   |   |-- valid/                  # immagini per scelta dei parametri/checkpoint
|   |   `-- test/                   # immagini per valutazione finale
|   |-- processed/                  # futuri indici filtrati e coppie generate
|   `-- pretrained/                 # due modelli Keras .h5 originali, esclusi da Git
|-- src/
|   `-- card_suit_duel/
|       |-- __init__.py
|       |-- train.py                # segnaposto: training non ancora implementato
|       |-- data/                   # codice per caricamento, coppie e Query
|       |-- logic/                  # programma ProbLog del duello
|       `-- models/                 # reti PyTorch e integrazione DeepProbLog
|-- scripts/                        # futuri comandi per download e preparazione
|-- tests/                          # test software; nessuna immagine del dataset
|-- notebooks/                      # esplorazione e studio
|-- results/                        # metriche, grafici e checkpoint locali
`-- docs/                           # traccia, documentazione e slide
```

Ogni cartella strutturale contiene un TXT con obiettivo e uso. Le sottocartelle originali del dataset (es. `ace of clubs`) restano inalterate. `.git`, `.venv` e `.idea` non sono mostrate nell'albero: sono rispettivamente metadati Git, ambiente Python locale e configurazione IDE.

Il precedente contenitore `CardSeedDuel/` e stato riunito alla radice Git: `configs`, `data`, `src` e `requirements.txt` si trovano ora insieme a README, scripts e tests. `cards.csv` e stato spostato dal package sorgente a `data/raw/`. Le immagini prima in `tests/test/` sono ora in `data/raw/test/`.

Era presente anche un secondo repository Git dentro `CardSeedDuel/`, senza commit ma con file staged. I suoi metadati sono conservati in `.local-backup-cardseedduel/.git`, esclusi da Git, per evitare di perdere quello stato. Il repository operativo e quello alla radice `Neurosymbolic_project/`, gia collegato a `https://github.com/gitCiccio/CardSeedDuel.git`; nessun commit o push e stato eseguito dal riordino.

## Dataset: origine, train, valid e test

Fonte: [Cards Image Dataset - Classification, gpiosenka su Kaggle](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification).
Per ricostruire i dati locali, scaricare ed estrarre l'archivio da Kaggle; collocare `train/`, `valid/`, `test/` e `cards.csv` in `data/raw/`. Gli eventuali due file `.h5` possono essere conservati in `data/pretrained/`, ma non sono necessari. Lo script automatico di download e ancora da implementare.

| Split | Uso | Immagini locali originali |
|---|---|---:|
| `data/raw/train/` | Apprendere i pesi con le coppie di training | 7624 |
| `data/raw/valid/` | Scegliere iperparametri, epoche e checkpoint senza aggiornare i pesi | 265 |
| `data/raw/test/` | Misurare le prestazioni finali dopo aver fissato le scelte | 265 |

Questi numeri precedono il filtro dei semi. Ogni split contiene le classi originali delle singole carte, inclusi Quadri e Joker. Mantenere raw intatto e applicare il filtro in fase di preparazione.

**Anomalia rilevata:** `cards.csv` ha 8155 righe e dichiara 7625 esempi di training, ma `train/ace of clubs/output` non esiste. Le immagini effettivamente presenti sono 8154. Il CSV e conservato senza modifiche: il futuro loader dovra validare i percorsi e segnalare/scartare esplicitamente quella riga. Non confondere questa anomalia con una coppia di training mancante.

Generare coppie solo fra immagini dello stesso split, preservando la separazione originaria. Controllare anche eventuali duplicati delle immagini fra split prima degli esperimenti. Riutilizzare le medesime coppie per DeepProbLog e baseline. Fissare validation e test; estrarre dal training sottoinsiemi riproducibili, preferibilmente annidati, al 20/50/100. Non usare test per il tuning.

## Cosa sono i due file 14card e 53cards?

Sono **modelli neurali Keras gia addestrati in formato HDF5 (.h5)**, non dataset, script o file necessari a DeepProbLog. Dalla configurazione incorporata nei file sono stati verificati input RGB `200 x 200` e ultimo layer Dense con softmax:

| File in `data/pretrained/` | Uscite verificate | Interpretazione |
|---|---:|---|
| `14card types-14-(200 X 200)-94.61.h5` | 14 | Classificatore dei tipi/valori delle carte, coerente con i 13 valori e Joker; l'ordine delle etichette deve essere verificato prima di un eventuale uso |
| `53cards-53-(200 X 200)-100.00.h5` | 53 | Classificatore delle singole carte (52 combinazioni valore-seme e Joker); verificare il mapping delle etichette prima di usarlo |

Le percentuali nei nomi sono indicazioni del distributore: non sono misure ottenute dal vostro progetto ne garanzie di prestazione. Nessuno dei due produce direttamente i tre semi o i tre risultati del duello. Il modello a 14 classi, classificando il valore, non risolve il riconoscimento del seme. I file Keras non si caricano direttamente come moduli PyTorch: per il progetto partiremo da reti PyTorch dedicate. Non sono stati eseguiti questi modelli; e stata ispezionata la loro configurazione.

## Ripartizione iniziale del lavoro

1. Dati: validazione CSV, filtro semi, preprocessing e coppie riproducibili.
2. DeepProbLog: rete dei semi, regole e pipeline di training/inferenza.
3. Baseline: modello a due immagini con predizione diretta dell'esito.
4. Esperimenti: protocollo comune, metriche, tempi, grafici e raccolta risultati.

Integrazione, revisione e presentazione sono condivise. Lo scheletro attuale documenta l'organizzazione: dipendenze, training, valutazione e download automatico devono ancora essere implementati e verificati.


## Installazione ed esecuzione della preparazione dati

Richiede Python 3.11 o superiore (compatibile con Pillow 13.0.0). Nella radice del repository, creare un ambiente
virtuale con un interprete valido e attivarlo. Quindi:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/inspect_dataset.py
```

`requirements.txt` richiama l'installazione editable del progetto; dipendenze e
package dentro `src/` sono dichiarati in `pyproject.toml`. Non occorre impostare
PYTHONPATH o Sources Root per eseguire il codice dopo l'installazione nello stesso
ambiente. Installare nuovamente se cambiano dipendenze o metadati del package.

È disponibile anche `card-inspect --project-root /percorso/del/progetto`.
Senza l'opzione, il comando usa la directory corrente come radice.

Lo script in `scripts/` è solo il punto di ingresso. Le funzioni riutilizzabili
sono in `card_suit_duel.data.inspection`: ricevono percorsi espliciti e non
configurano il logger. `run_inspection()` restituisce il report senza scriverlo;
`save_report()` lo salva. I dati originali restano invariati.

Per il contratto dei metadati vedere [docs/dataset_schema.md](docs/dataset_schema.md).
Per il dettaglio della riorganizzazione vedere [docs/review_fixes.md](docs/review_fixes.md).
