# Approccio A — DeepProbLog (`feature/deepproblog`)

Questo documento descrive cosa è stato implementato nel branch `feature/deepproblog` per l'**Approccio A (neuro-simbolico con DeepProbLog)**: il programma logico, l'adattatore tra le coppie preparate e DeepProbLog, il trainer, lo script di training e i test.

Il branch parte da `feature/baseline` e riusa i suoi mattoni condivisi (`CardEncoder`, `SuitNet`, `CardTensorCache`, metriche e blocco hardware), così A e B hanno **la stessa capacità visiva, lo stesso preprocessing e le stesse metriche**.

---

## 1. Obiettivo dell'Approccio A

Date due immagini di carte $(I_1, I_2)$, il modello predice l'esito del duello tra le 3 classi di `OUTCOME_CLASSES` (`p1_win`, `p2_win`, `draw`), come nell'Approccio B, con la stessa supervisione: **solo l'esito** `(img1, img2, outcome)`, mai il seme delle singole carte.

La differenza rispetto a B:
- **la percezione** (immagine → seme) è appresa da una rete neurale, `SuitNet`;
- **la regola del gioco** ($\text{Hearts} > \text{Clubs} > \text{Spades} > \text{Hearts}$, semi uguali $\to$ `draw`) è **scritta in ProbLog e non viene appresa**.

DeepProbLog calcola la probabilità esatta dell'esito combinando le distribuzioni sui semi prodotte dalla rete con le regole, e il gradiente della loss sull'esito risale fino a `SuitNet`.

---

## 2. Mappa dei file del branch

```text
CardSeedDuel/
|-- src/card_suit_duel/logic/
|   |-- __init__.py                  # PROGRAM_PATH, SUIT_NETWORK_NAME ("suit_net"), DUEL_PREDICATE ("duel")
|   `-- card_duel.pl                 # Programma ProbLog: predicato neurale suit/2 + regole note
|-- src/card_suit_duel/models/
|   |-- deepproblog_adapter.py       # CardImages (tensor source), DuelQueries (query), build_deepproblog_model, register_split
|   `-- deepproblog_trainer.py       # train_one_epoch_deepproblog, evaluate_deepproblog, evaluate_suit_predictions, fit_deepproblog
|-- scripts/
|   `-- train_deepproblog.py         # CLI gemella di train_baseline.py (--evaluate-test OFF di default)
|-- tests/
|   `-- test_deepproblog.py          # 6 test: regole vs oracolo, query, inferenza esatta, training, diagnostica semi
|-- docs/
|   `-- DEEPPROBLOG_APPROACH_A.md    # Questo documento
`-- pyproject.toml                   # Nuovo extra "neurosymbolic": deepproblog==2.1.0, problog==2.3.0
```

Nessun file esistente di `data/`, `game/` o dei moduli della baseline è stato modificato. `models/__init__.py` **non** esporta i moduli DeepProbLog, così la baseline continua a funzionare anche senza `deepproblog` installato.

---

## 3. Programma logico (`src/card_suit_duel/logic/card_duel.pl`)

```prolog
nn(suit_net, [X], S, [hearts, clubs, spades]) :: suit(X, S).

beats(hearts, clubs).
beats(clubs, spades).
beats(spades, hearts).

outcome(S, S, draw).
outcome(S1, S2, p1_win) :- beats(S1, S2).
outcome(S1, S2, p2_win) :- beats(S2, S1).

duel(X, Y, O) :- suit(X, S1), suit(Y, S2), outcome(S1, S2, O).
```

- `suit/2` è l'**unico predicato neurale**: `SuitNet` restituisce una distribuzione sui 3 semi, nello stesso ordine di `SUIT_CLASSES`.
- `beats/2` e `outcome/3` replicano esattamente l'oracolo `card_suit_duel.game.oracle.determine_outcome`. Un test lo verifica su tutte le 9 combinazioni di semi.
- I nomi degli atomi coincidono con `SUIT_DOMAIN` e `OUTCOME_DOMAIN`.

La probabilità che il programma calcola per una coppia è:

$$P(\text{draw}) = \sum_s p_1(s)\,p_2(s), \qquad P(\text{p1\_win}) = \sum_{(a,b)\,:\,\text{beats}(a,b)} p_1(a)\,p_2(b)$$

e simmetricamente per `p2_win`. Un test confronta questi valori, calcolati a mano, con quelli dell'`ExactEngine`.

---

## 4. Rete neurale

È la `SuitNet` già presente in `encoder.py` (branch `feature/baseline`), fedele a `MNIST_Net` del **Lab 03**:

`CardEncoder` (normalizzazione `[-1, 1]` → `Conv(3,6,5)` → `MaxPool` → `ReLU` → `Conv(6,16,5)` → `MaxPool` → `ReLU` → `AdaptiveAvgPool(4×4)` = 256 feature) → `256 → 120 → ReLU → 84 → ReLU → 3` → `Softmax(dim=1)`.

L'encoder è la **stessa classe** usata da `BaselineDuelNet` nell'Approccio B. La rete è registrata in DeepProbLog con `Network(SuitNet(), "suit_net", batching=True)` e ottimizzata con Adam.

---

## 5. Adattatore dati (`deepproblog_adapter.py`)

### 5.1 `DuelQueries`
Converte le `CardPair` caricate con `load_pairs` (pipeline del PR #18) in query DeepProbLog:

```text
duel(tensor(train(a)), tensor(train(b)), p1_win)    con sostituzione {a: 17, b: 342}
```

- Gli indici delle immagini sono passati **tramite sostituzione**, come nell'esempio MNIST-addition di DeepProbLog. Tutte le query con lo stesso esito hanno quindi la stessa struttura e l'`ExactEngine` (con `cache=True`) compila **un solo circuito aritmetico per esito**, cioè 3 compilazioni in tutto.
- Rifiuta le coppie che non appartengono allo split dichiarato (nessuna immagine di valid o test può finire nel train).
- `cards` restituisce le carte uniche nell'ordine di prima comparsa: la posizione coincide con l'indice usato nelle query.

### 5.2 `CardImages`
È la *tensor source* registrata con `model.add_tensor_source(split, ...)`. Risolve `tensor(train(17))` nel tensore `[3, H, W]` della carta 17, letto dalla **stessa `CardTensorCache` dell'Approccio B** (stessa decodifica, stesso resize).

### 5.3 `build_deepproblog_model` e `register_split`
- `build_deepproblog_model(SuitNet(), lr=...)` crea `Network`, ottimizzatore Adam, `Model` dal file `.pl` e `ExactEngine` con cache.
- `register_split(model, dataset, cache)` registra la tensor source di uno split e la **pre-carica in RAM**, restituendo i secondi impiegati. In questo modo i tempi di training non includono la decodifica dei JPEG, come nella baseline.

---

## 6. Trainer (`deepproblog_trainer.py`)

| Funzione | Cosa fa |
|---|---|
| `train_one_epoch_deepproblog` | Batch mescolati con un `random.Random(seed)` proprio → `model.solve(batch)` (inferenza esatta) → `semiring.cross_entropy` sull'esito vero (fa anche il `backward`) → `optimizer.step()`. Restituisce loss media, numero di batch e secondi. |
| `evaluate_deepproblog` | Per ogni coppia interroga `duel(A, B, O)` con `O` variabile, prende $\arg\max_O P(O)$ e calcola loss (NLL dell'esito vero), accuracy, Macro-F1 e matrice di confusione con `compute_classification_metrics`, la stessa funzione della baseline. |
| `evaluate_suit_predictions` | **Solo diagnostica**: confronta l'argmax di `SuitNet` con il seme vero delle carte. Le etichette dei semi non vengono mai usate né per il training né per scegliere il checkpoint. Vedi §7. |
| `fit_deepproblog` | Loop sulle epoche con la **stessa regola di checkpoint della baseline**: migliore Macro-F1 di valid, a parità la loss di valid più bassa. Ricarica il checkpoint migliore alla fine. |

Differenza rispetto alla baseline: per il train viene registrata solo la loss, non accuracy e F1. Calcolarle richiederebbe una seconda inferenza con l'esito variabile su ogni coppia di train, raddoppiando il costo dell'epoca.

---

## 7. Simmetria ciclica dei semi

La regola è **invariante per rotazione ciclica** dei nomi dei semi: se ogni seme viene rinominato con `hearts → clubs → spades → hearts`, `beats/2` resta identico. Poiché la rete vede solo gli esiti, può imparare una percezione perfetta ma con i nomi ruotati e ottenere comunque il 100% sulle partite.

Per questo `evaluate_suit_predictions` riporta:
- `accuracy` e Macro-F1 diretti (nome del seme previsto = nome vero);
- `accuracy_up_to_rotation`: la migliore accuratezza tra le 3 rotazioni cicliche;
- `best_rotation_shift`: quale rotazione la rete ha imparato (0 = nomi corretti).

È un punto da discutere nelle slide: DeepProbLog impara i concetti *a meno delle simmetrie della teoria*.

---

## 8. Script e comandi

### Installazione delle dipendenze
```powershell
.venv\Scripts\python.exe -m pip install -e ".[neurosymbolic]"
```
DeepProbLog all'import avvisa che `ApproximateEngine` non è disponibile perché manca PySwip: non serve, il progetto usa solo `ExactEngine`.

### Test unitari
```powershell
.venv\Scripts\python.exe -B -m unittest tests.test_deepproblog -v
```

### Tuning / sviluppo (solo `train` e `valid`, test NON toccato)
```powershell
.venv\Scripts\python.exe scripts\train_deepproblog.py --percentage 20 --image-size 64 --epochs 25 --batch-size 64 --lr 0.001 --seed 0
```

### Prova veloce (pochi batch)
```powershell
.venv\Scripts\python.exe scripts\train_deepproblog.py --percentage 20 --epochs 1 --max-train-batches 20 --max-eval-batches 2 --seed 0
```

### Valutazione finale (una sola volta, a configurazione concordata con l'Approccio B)
```powershell
.venv\Scripts\python.exe scripts\train_deepproblog.py --percentage <20|50|100> --image-size 64 --epochs <N> --batch-size 64 --lr 0.001 --seed <0|1|2> --evaluate-test
```

Ogni run scrive in `results/runs/deepproblog/<timestamp>_p<perc>_<id>/` (cartella ignorata da git):
- `best_model.pt`: pesi di `SuitNet` al checkpoint migliore, aggiornato durante il training ogni volta che la validazione migliora;
- `checkpoints/epoch_NNN.pt`: checkpoint completo scritto a fine di ogni epoca (pesi, stato di Adam, stato dei generatori casuali, `history` e migliore finora), salvato in modo atomico;
- `summary.json`: stessi campi della baseline (`hardware`, `cache_seconds`, `cache_details`, `epochs_completed`, `best_epoch`, `best_valid_macro_f1`, `best_valid_accuracy`, `best_valid_ms_per_pair`, `best_valid_per_suit_combination`, `total_training_seconds`, `trainable_parameters`, `history`, `test`; ogni valutazione include `ms_per_pair` e `per_suit_combination` sulle 9 combinazioni di semi) più `engine`, `checkpoint_dir`, `resumed_from`, `valid_suit_diagnostics` e `test_suit_diagnostics`.

### Ripresa di una run interrotta
Si riparte dall'ultimo checkpoint, ripassando **gli stessi flag** della run originale e il numero totale di epoche desiderato; la run continua nella stessa cartella:
```powershell
.venv\Scripts\python.exe scripts\train_deepproblog.py --percentage 20 --image-size 64 --epochs 25 --batch-size 64 --lr 0.001 --seed 0 --resume-from results\runs\deepproblog\<run>\checkpoints\epoch_010.pt
```
La ripresa riproduce la stessa traiettoria di una run senza interruzioni (verificato in `tests/test_deepproblog.py`).

---

## 9. Test (`tests/test_deepproblog.py`)

| Test | Cosa verifica |
|---|---|
| `test_rules_match_oracle_on_all_nine_suit_combinations` | Con ProbLog puro (semi fissati come fatti), `duel/3` dà lo stesso esito di `determine_outcome` in tutte le 9 combinazioni. |
| `test_queries_reference_split_source_and_share_circuit_structure` | Forma delle query, indici della tensor source e struttura condivisa tra query con lo stesso esito. |
| `test_rejects_pairs_from_another_split` | Coppie di split diversi o lista vuota vengono rifiutate. |
| `test_exact_inference_matches_closed_form_outcome_probabilities` | Le probabilità dell'`ExactEngine` coincidono (5 decimali) con la formula del §3. |
| `test_fit_deepproblog_reduces_validation_loss` | Su 9 coppie sintetiche il training riduce la loss, salva il checkpoint e conta i parametri di `SuitNet`. |
| `test_rotated_suit_naming_is_recognized` | La diagnostica riconosce una percezione corretta ma con nomi ruotati. |

L'intera suite del progetto (240 test) passa.

---

## 10. Stato e prossimi passi

**Fatto:** programma logico, adattatore, trainer, script, test e uno smoke run sui dati reali (20%, 20 batch di train, 2 batch di valid), che gira dall'inizio alla fine e scrive il `summary.json`.

**Costo computazionale (stima dallo smoke run, CPU condivisa con un altro training):**
- circa 12 ms per coppia in training e circa 7 ms per coppia in valutazione;
- quindi circa 1,5 minuti per epoca al 20% e circa 6 minuti per epoca al 100%;
- con 25 epoche: circa 40 minuti per un run al 20% e circa 2,5 ore per uno al 100%; la griglia completa 20/50/100 × 3 seed richiederebbe circa 14 ore.

Sono stime grezze da misurare di nuovo a CPU libera. Il tempo di training più alto rispetto alla baseline è comunque un risultato da riportare nel confronto A vs B.

**Da fare:**
1. Concordare con Simone, che ha in carico l'Approccio A, questa implementazione e la divisione del lavoro restante.
2. Fissare insieme all'Approccio B risoluzione (64×64), batch, lr ed epoche, scegliendo solo sul validation.
3. Eseguire la griglia 20% / 50% / 100% × seed 0, 1, 2 (solo train e valid).
4. Eseguire `--evaluate-test` **una sola volta** a configurazione bloccata, per A e B.
