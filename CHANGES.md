# Changes - ultimo commit e consegna al team

Data: 17 novembre 2026. Questo documento descrive il **commit corrente**,
come verificarlo e su cosa può iniziare a lavorare il team. La storia
completa delle decisioni F2.1–F2.10 è in [CHANGELOG.md](CHANGELOG.md).

## Sintesi per la revisione

Il commit completa il caricamento neurale delle coppie e aggiunge un comando
per riprodurre l’intera preparazione dati. Gli indici approvati restano
immutati: due nuove preparazioni hanno ricostruito gli stessi nove CSV.
Il team può partire dagli stessi input per sviluppare baseline e DeepProbLog.
Non sono inclusi modelli addestrati né una verifica già eseguita della loro
integrazione. Nessun commit o push viene effettuato automaticamente.

## Perimetro del commit corrente

### Codice nuovo

| Percorso | Responsabilità |
|---|---|
| src/card_suit_duel/data/path_checks.py | Unico controllo condiviso di radice, risoluzione, contenimento ed esistenza |
| src/card_suit_duel/data/image_io.py | Lettura RGB con chiusura del sorgente e immagine indipendente in memoria |
| src/card_suit_duel/data/preprocessing.py | Conversione float32 [0,1], dimensioni native e resize configurabile |
| src/card_suit_duel/data/pairs/pair_dataset.py | Lettura lazy del duello con due immagini, target e ID diagnostici |
| src/card_suit_duel/data/pairs/pair_loader.py | Batch, seed, shuffle soltanto train, conservazione dell’ultimo batch |
| scripts/check_pair_loaders.py | Verifica del caricamento a partire dal manifest F2.8 |
| src/card_suit_duel/data/deduplication/manual_review.py | Applicazione delle decisioni manuali legate al fingerprint |
| src/card_suit_duel/data/deduplication/exporting.py | Builder/export condivisi dei risultati di deduplicazione |
| src/card_suit_duel/data/deduplication/pipeline.py | Deduplicazione riutilizzabile con replay della revisione |
| src/card_suit_duel/data/preparation.py | Orchestrazione delle fasi esistenti dal raw ai sottoinsiemi |
| scripts/prepare_dataset.py | Comando principale della preparazione |

### Moduli aggiornati

- inspection, validation, validation_suite e fingerprints riutilizzano
  i controlli pubblici dei percorsi, senza duplicare la logica filesystem.
- Il vecchio deduplication_dataset.py riutilizza i builder condivisi,
  mantenendo la sua interfaccia CLI.
- pyproject.toml espone l’extra opzionale neural con torch/torchvision
  verificati. requirements.txt mantiene l’installazione base del package.
- .gitignore rende work_sessions versionabile; pesi, ambienti e backup
  restano esclusi.
- README e configs/config.txt documentano la preparazione effettivamente
  disponibile. Il contratto dei metadati è allineato agli import attuali.

### Configurazioni e test nuovi

- configs/preparation.json: percorsi, quantità e seed del comando unico.
- configs/duplicate_review_decisions.json: evidenza della revisione visiva
  e rappresentante esplicito; non è una scelta automatica di etichette.
- tests/test_image_io.py: 15 test dei percorsi e della lettura.
- tests/test_preprocessing.py: 9 test delle trasformazioni.
- tests/test_pair_loading.py: 23 test Dataset/DataLoader.
- tests/test_preparation.py: 7 test orchestrazione e revisione.

Sono **54 test nuovi rispetto ai 169 precedenti**, per **223 test totali**.
Nell’ambiente verificato sono stati eseguiti anche tutti i test neural.

### Documentazione e risultati da includere

- CHANGELOG.md e CHANGES.md.
- docs/F2_9_AMBIENTE_NEURALE.md e F2_9_SCALETTA_E_CONTRATTO.md.
- docs/F2_9_VERIFICA_COMPLETA.json.
- docs/F2_10_RIPRODUCIBILITA.md e F2_10_VERIFICA_LOCALE.json.
- Recap in work_sessions, in particolare quelli del 16 e 17 novembre.
- Report di results/pipeline_output/loading_check e run della preparazione
  usati per le evidenze. I report parziali non vanno presentati come completi.

docs/review_fixes.md era già nello staging: revisionarlo come documento
precedente, non considerarlo un nuovo cambiamento dell’algoritmo dei loader.
Il resto di F2.1–F2.8 già nel branch non deve essere descritto come
reimplementato in questo commit.

## Contratto di consegna ai due approcci

| Elemento | Contratto |
|---|---|
| Immagini | File originali in data/raw, riferimenti dagli indici approvati |
| Duello | Due immagini distinte dello stesso split, ordine dei giocatori significativo |
| Classi semi | hearts=0, clubs=1, spades=2 |
| Classi esiti | p1_win=0, p2_win=1, draw=2 |
| Immagine preprocessata | float32 [3,224,224] in [0,1] nella configurazione iniziale |
| Batch | first_image e second_image [B,3,224,224], target long [B] |
| Diagnostica | pair_id, first_image_id, second_image_id separati dalle feature |
| Valutazione | Valid/test fissi, senza shuffle o scarto dell’ultimo batch |

Usare outcome_to_index e gli altri mapping del package, senza ridefinirli.
Il preprocessing è sostituibile: una rete preaddestrata può richiedere un
contratto diverso, da concordare e documentare per entrambi gli approcci.
Non usare i semi reali dei metadati come risposta già nota al classificatore.
Eventuale supervisione aggiuntiva sui semi deve essere una scelta sperimentale
esplicita. Le prestazioni sui duelli non provano da sole l’identificazione
semantica dei nomi dei semi in DeepProbLog.

Per la baseline, usare CardPairDataset e build_pair_loader. Per DeepProbLog,
F3A deve implementare tensor source e Dataset.to_query riusando gli stessi
image_id/pair_id, outcome e trasformazioni condivise. Il suo DataLoader
specifico non è automaticamente sostituito dal loader PyTorch della baseline.

## Indici pronti per gli esperimenti

Percorsi relativi alla radice del repository:

```text
Immagini approvate:
results/pipeline_output/deduplication/20261114T180134Z_manual_review/deduplicated_images.csv

Manifest esperimenti:
results/pipeline_output/subsets/20261115T164052Z_742a2129/summary.json

Train:
results/pipeline_output/subsets/20261115T164052Z_742a2129/20/train_pairs.csv
results/pipeline_output/subsets/20261115T164052Z_742a2129/50/train_pairs.csv
results/pipeline_output/subsets/20261115T164052Z_742a2129/100/train_pairs.csv

Valutazione fissa:
results/pipeline_output/pairs/20261115T105737Z_b3f5a3e3/valid_pairs.csv
results/pipeline_output/pairs/20261115T105737Z_b3f5a3e3/test_pairs.csv
```

Train: 6000/15000/30000 coppie, incluse l’una nell’altra. Valid/test:
3000 coppie ciascuno. Le immagini train uniche coinvolte sono rispettivamente
4989/5651/5681: percentuale delle coppie e delle immagini non coincidono.

## Installazione e verifica del compagno

Ogni compagno crea il proprio ambiente: non copiare la .venv locale.
Configurazione CPU di riferimento, dalla radice del repository:

```text
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -e ".[neural]"
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_*.py"
.venv\Scripts\python.exe scripts/prepare_dataset.py --config configs/preparation.json
```

Il comando ricostruisce indici e report dal raw già presente. Nuovi run
non sovrascrivono quelli condivisi. Confrontare reproducibility.json con
la verifica locale, non i timestamp dei report operativi.

Fingerprint semantico di riferimento:

```text
6880fe9dfd491615b199686b0f29318fa8eff4867dc1bd3835d69ad7766e872d
```

Per la prova completa del loader, usare lo stesso interprete:

```text
python scripts/check_pair_loaders.py --subset-manifest "results/pipeline_output/subsets/20261115T164052Z_742a2129/summary.json" --image-index "results/pipeline_output/deduplication/20261114T180134Z_manual_review/deduplicated_images.csv" --dataset-root "data/raw" --output-root "results/pipeline_output/loading_check" --percentage 20 --batch-size 32 --seed 42 --max-batches 1000
```

Ripetere con percentage 50 e 100. In PyCharm lo script path è il file .py,
i parametri vanno in Script parameters e la working directory è la radice.
Il controllo completo legge le immagini; non allena una rete.

## Evidenze dell’ultimo commit

- 223 test superati; pip check senza conflitti.
- Tutte le coppie train/valid/test caricate nelle tre condizioni.
- Ultimi batch incompleti mantenuti, target e ordine dei giocatori verificati.
- Due preparazioni complete dal raw riproducono gli stessi nove CSV e
  lo stesso reproducibility.json, coincidenti anche con gli indici approvati.
- Gli originali e i run precedenti non sono modificati.

Run completi loader: 20261116T231929Z_f1190810 (20%),
20261116T232034Z_d30b523f (50%), 20261116T232221Z_e49c969d (100%).
Run riproducibilità: 20261116T232600Z_2dc1d17f e
20261116T232624Z_4d5f1014. Questi ID sono UTC, anche quando il log locale
riporta già il 17 novembre.

## Via libera al team e condizioni ancora aperte

**Via libera all’implementazione dei due modelli sugli indici condivisi.**
Non è una dichiarazione che i due modelli siano già integrati o che il
preprocessing definitivo sia concordato.

- F2.9: criterio dei tensori e batch verificato localmente; stato remoto
  non aggiornato dall’assistente perché la richiesta del connettore è stata rifiutata.
- F2.10: comando e riproducibilità locale completati; mancano prova documentata
  di un compagno su clone pulito e caricamento degli stessi esempi nei due modelli.
- F3A: adattatore Query/tensor source e regole ProbLog da integrare.
- F3A/F3B: architettura, pesi e preprocessing definitivo da concordare.
- Il confronto percettivo dei quasi-duplicati non è stato eseguito.
- Le scritture non sono atomiche: una cartella incompleta non è un run approvato.

Il compagno che riproduce registri sistema, versioni, run ID e confronto
del fingerprint nella sessione o nell’issue #17. Non sostituire questa prova
con una seconda esecuzione sulla stessa macchina dell’autore.

## Preparazione pratica del commit

Prima di approvare lo staging, escludere i due pesi .h5 preesistenti,
circa 135 MB ciascuno. Erano già staged: gitignore non li rimuove.
Questi comandi rimuovono soltanto lo staging, non i file locali:

```text
git restore --staged -- "data/pretrained/14card types-14-(200 X 200)-94.61.h5" "data/pretrained/53cards-53-(200 X 200)-100.00.h5"
```

Ricontrollare i file AM: alcuni erano stati aggiunti vuoti e poi modificati;
occorre includerne la versione finale nello staging. Esempio di selezione
del perimetro dati, senza aggiungere ambienti o pesi:

```text
git add .gitignore README.md CHANGELOG.md CHANGES.md pyproject.toml configs scripts src tests docs work_sessions results/pipeline_output/loading_check results/pipeline_output/preparation
git diff --cached --check
git diff --cached --stat
```

Revisionare l’elenco prima di committare, soprattutto eventuali altri file
già nello staging. Non è stato eseguito alcun reset, commit o push.

## Messaggio di commit proposto

```text
Completa i loader delle coppie e integra la preparazione riproducibile

- Centralizza i controlli dei percorsi e verifica il preprocessing
- Controlla tutti i batch train, validation e test al 20/50/100%
- Aggiunge comando principale, configurazione e revisioni manuali versionate
- Verifica due preparazioni complete con gli stessi indici e manifest
- Documenta decisioni F2.1–F2.10 e consegna operativa al team

Verifica: 223 test superati.
```

## Testo breve da condividere dopo il push

La preparazione dati e il loader delle coppie sono disponibili e verificati.
Usiamo gli stessi train annidati 20/50/100% e valid/test fissi. Nel commit
trovate configurazione, comando di ricostruzione, 223 test e documentazione
delle decisioni. Potete iniziare i due modelli con mapping e ID condivisi.
Resta da concordare il preprocessing definitivo; DeepProbLog richiede il
suo adattatore Query/tensor source. Serve inoltre una verifica da clone
di un compagno per completare i criteri esterni di F2.10.

Il testo è preparato per il responsabile: non è stato inviato al team.


## Elenco esatto e ultimo run condiviso

L’elenco completo dei file da includere, ricavato dallo stato Git al momento
della consegna, è in [docs/COMMIT_FILES.txt](docs/COMMIT_FILES.txt). Include
anche i file dentro le cartelle dei report, non soltanto il nome della directory.
Non include i due .h5 già staged. Prima di usarlo rimuovere quei pesi dallo
staging con il comando riportato sopra, poi dalla radice:

```text
git add --pathspec-from-file=docs/COMMIT_FILES.txt
git diff --cached --check
git diff --cached --stat
```

L’ultimo run eseguito dal responsabile è
`20261116T234548Z_35065d09`. Il suo reproducibility.json coincide con la
verifica precedente e tutti i nove CSV corrispondono agli hash dichiarati.
Questo è il run indicato nel messaggio di consegna: leggere il suo
summary.json per ottenere i percorsi invece di scegliere un run per data
o riscrivere gli ID delle sottocartelle.

Messaggio pronto ed esempio di caricamento:
[docs/MESSAGGIO_TEAM_DATASET.md](docs/MESSAGGIO_TEAM_DATASET.md).
Il testo non è stato inviato dal programma a nessun collega.
L’inventario descrive lo stato corrente: eventuali file aggiunti dopo
la sua creazione devono essere revisionati e aggiunti separatamente.
