# Verifica dei campi dei run per il confronto A vs B

Data: 26/11/2026 · Branch: `feature/deepproblog` (`3338e66`, allineato con `main` `7dc39da`)

Scopo: controllare che i `summary.json` prodotti da `scripts/train_baseline.py` (Approccio B) e
`scripts/train_deepproblog.py` (Approccio A) contengano tutto ciò che serve al futuro script di
confronto, **prima** di lanciare la griglia finale 20/50/100% × seed 0, 1, 2.

Run esaminati:

- A: `results/runs/deepproblog/20261126T134942Z_p20_90d3f8af` (20%, 30 epoche, seed 0, solo valid).
- B: `results/runs/baseline_seeds/20261125T*` (9 run, 20/50/100% × seed 0, 1, 2, 25 epoche, solo valid),
  confrontati anche con il codice attuale di `train_baseline.py` su `main`.

---

## 1. Esito in breve

| Requisito della consegna | Campi disponibili | A | B |
|---|---|---|---|
| Accuracy e F1 sul **test** | `test.accuracy`, `test.macro_f1` (+ precision, recall, per classe, matrice di confusione) | ✅ | ✅ |
| Ripetere con 20/50/100% | `training_percentage` | ✅ | ✅ |
| Tempo di training | `total_training_seconds`, `history[].train.elapsed_seconds` | ✅ | ✅ |
| Efficienza computazionale | `ms_per_pair` (valid/test), `trainable_parameters`, `cache_seconds`, `hardware` | ✅ | ✅ |
| Validation e test invariati | stessi CSV di default nei due script (`valid_pairs.csv`, `test_pairs.csv`, 3000 coppie ciascuno) | ✅ | ✅ |
| Discussione delle differenze | `history` per epoca, `per_suit_combination`, diagnostica dei semi (solo A) | ✅ | ✅ |

**Conclusione:** con il codice attuale entrambi gli script salvano tutto il necessario. **I 9 run della
baseline del 25/11 vanno però rifatti**: sono stati prodotti con una versione precedente del codice
(vedi §3) e con 25 epoche.

---

## 2. Campi del `summary.json` per singolo run

Campi presenti **in entrambi** (codice attuale):

| Gruppo | Campi |
|---|---|
| Identificazione | `approach`, `run_id`, `training_percentage`, `seed` |
| Configurazione | `image_size`, `batch_size`, `lr`, `epochs`, `epochs_completed` |
| Selezione del modello | `best_epoch`, `best_valid_macro_f1`, `best_valid_loss`, `best_valid_accuracy`, `best_valid_ms_per_pair`, `best_valid_per_suit_combination` |
| Costo | `total_training_seconds`, `trainable_parameters`, `cache_seconds`, `cache_details` |
| Hardware | `hardware` (CPU, thread di torch, versioni, GPU) |
| Curve | `history[]` con `train` e `valid` per epoca |
| Test | `test_evaluated`, `test` (stesse metriche di valid, con `ms_per_pair` e `per_suit_combination`) |

Solo in **A**: `engine`, `checkpoint_dir`, `resumed_from`, `valid_suit_diagnostics`,
`test_suit_diagnostics` (accuratezza dei semi diretta e `accuracy_up_to_rotation`).

Solo in **B**: `patience`, `stopped_early`.

Ogni valutazione (`history[].valid` e `test`) contiene: `loss`, `accuracy`, `macro_f1`,
`macro_precision`, `macro_recall`, `per_class`, `confusion_matrix`, `sample_count`,
`elapsed_seconds`, `ms_per_pair`, `per_suit_combination` (9 combinazioni ordinate di semi).

---

## 3. Problemi trovati e cosa fare

### 3.1 Run della baseline del 25/11 da rifare (bloccante)

I run in `results/runs/baseline_seeds/` sono precedenti a `cef1281` e **non hanno**
`ms_per_pair`, `per_suit_combination`, `best_valid_accuracy`, `epochs_completed`,
`patience`/`stopped_early`. In più usano 25 epoche, e quasi sempre il best epoch è tra 22 e 25.

Il run 100%/seed 1 è rimasto bloccato su una sola classe (Macro-F1 0,167) per 24 epoche e ha
iniziato a imparare solo all'ultima (0,46): non è un bug, ma va verificato con più epoche.

→ Rilanciare la griglia della baseline con il codice attuale e lo stesso numero di epoche di A.

### 3.2 Numero di coppie di training in A (risolto)

B salvava `history[].train.sample_count`, A no (solo `batch_count`). Serve per calcolare i
**ms per coppia in training**. Ora `train_one_epoch_deepproblog` salva anche `sample_count`,
quindi i nuovi run di A lo hanno. Per il run esplorativo del 26/11 si ricava da `training_percentage`:

| Percentuale | Coppie di training |
|---|---|
| 20% | 6000 |
| 50% | 15000 |
| 100% | 30000 |

Valid e test: 3000 coppie ciascuno.

### 3.3 Metriche di training diverse (minore)

`history[].train` di B ha anche accuracy e Macro-F1; quello di A ha solo la `loss`
(calcolarle richiederebbe un'inferenza in più). Le curve di training confrontabili sono quindi
quelle di **loss**; entrambe sono cross-entropy sull'esito, quindi sulla stessa scala.
Le curve di **valid** sono identiche nei due approcci.

### 3.4 Device diverso tra gli script (attenzione per l'altro PC)

A misura sempre su CPU (`collect_hardware_info("cpu")`), B usa `--device` (default `cpu`).
Se l'altro PC ha una GPU, **lanciare B con `--device cpu`**, altrimenti i tempi non sono confrontabili.
Su questa macchina entrambi risultano con 24 CPU logiche e `torch_num_threads = 12`.

### 3.5 Struttura delle cartelle diversa (per lo script di confronto)

- B multi-seed (`--seeds 0 1 2`): `results/runs/baseline/<run_id>/summary.json` aggregato +
  `seed_<N>/summary.json` per seed. Il sommario aggregato non contiene tutto (per esempio le
  `per_suit_combination` sul test), quindi conviene leggere i file `seed_<N>/summary.json`.
- A: una cartella per run in `results/runs/deepproblog/<run_id>/summary.json`, un seed per run
  (il seed è nel campo `seed`, non nel nome della cartella).

→ Lo script di confronto deve leggere i `summary.json` dei singoli seed di entrambi e
aggregare da sé (può riusare `aggregate_metric_runs`).

### 3.6 Run ripresi con `--resume-from` (minore)

`total_training_seconds` e `history` si sommano correttamente tra le sessioni, ma
`cache_seconds` e `hardware` si riferiscono solo all'ultima sessione. Se un run viene ripreso
su un'altra macchina i tempi si mescolano: evitarlo nei run finali.

---

## 4. Checklist per i run finali

- Stessa configurazione per A e B: `--image-size 64 --batch-size 64 --lr 1e-3 --epochs 30`.
- Entrambi gli script hanno `--epochs 5` di default e A ha `--seed 42`: **passare sempre i valori in modo esplicito**.
- Seed 0, 1, 2 per ogni percentuale (A: un comando per seed; B: `--seeds 0 1 2`).
- B con `--device cpu`; tutti i run sulla stessa macchina, possibilmente a PC libero.
- Prima la griglia **solo su valid**; `--evaluate-test` una sola volta, a configurazione bloccata.
- Copiare a mano le cartelle dei run (`results/runs/` è nel `.gitignore`).

Esempi:

```bash
.venv/Scripts/python -B scripts/train_deepproblog.py --percentage 20 --epochs 30 --seed 0
.venv/Scripts/python -B scripts/train_baseline.py --percentage 20 --epochs 30 --seeds 0 1 2 --device cpu
```

---

## 5. Riferimento: run esplorativo di A (20%, seed 0)

| Voce | Valore |
|---|---|
| Best Macro-F1 valid | 0,878 (epoca 30; 0,867 già all'epoca 18) |
| Minimo loss valid | 0,455 (epoca 24), poi overfitting sulla loss |
| Tempo di training | 1403 s per 30 epoche (~47 s/epoca, ~7,8 ms/coppia) |
| Inferenza su valid | ~4,5 ms/coppia |
| Accuratezza semi | 2% diretta, 93% a meno della rotazione ciclica |
| Parametri allenabili | 44.131 |

Il processo girava a priorità ridotta mentre il PC era in uso: i tempi sono indicativi.
