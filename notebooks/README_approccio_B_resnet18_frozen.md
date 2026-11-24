# Approccio B con ResNet18 congelata — esperimento preliminare (Simone)

Notebook: [`approccio_B_resnet18_frozen_colab.ipynb`](approccio_B_resnet18_frozen_colab.ipynb)

[![Apri in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gitCiccio/CardSeedDuel/blob/experiment/resnet18-frozen-colab/notebooks/approccio_B_resnet18_frozen_colab.ipynb)

## Perché questo esperimento

Prima di scegliere il backbone comune ad A (DeepProbLog) e B (baseline neurale) volevo capire se una ResNet18
pre-addestrata su ImageNet rende il problema **troppo facile**. Se la sola baseline neurale arrivasse quasi al 100%
già con il 20% delle coppie, il confronto con DeepProbLog sull'accuratezza sarebbe schiacciato verso l'alto
(effetto soffitto) e il vantaggio del ragionamento simbolico non si vedrebbe.

Il notebook usa il caso più semplice ed economico: backbone **congelato** e training della sola testa.
Non è il modello finale proposto, ma serve a decidere dove investire.

## Come eseguirlo (nessuno zip, nessun Drive)

1. Aprire il link "Apri in Colab" qui sopra.
2. *Runtime → Cambia tipo di runtime → GPU (T4)*.
3. *Runtime → Esegui tutto*. Servono circa 3-4 minuti: ~30 s di clone, ~6 s per caricare le immagini, ~15 s per estrarre
   le feature e ~2,5 minuti per i 9 training.

La prima cella clona questo branch del repo pubblico: immagini (`data/raw/`) e coppie sono già versionate in `main`.
Lo zip `CardSeedDuel_colab.zip` usato nel primo run conteneva esattamente gli stessi dati, verificati file per file
(CSV identici a parte i fine riga, immagini identiche byte per byte), quindi non serve più.
Il notebook funziona anche in locale, se lo si apre da dentro il repo: trova da solo la radice.

Gli output salvati nel notebook vengono dal run del **17/11/2026 su Colab, Tesla T4**
(Python 3.13, torch 2.14.1+cu130, torchvision 0.29.1). In quel run i dati arrivavano dallo zip.
Rispetto a quel run sono cambiate solo le celle che leggono i dati (2, 3 e 7); modello e training sono invariati.

## Dati

Run finale della pipeline del PR #18: `results/pipeline_output/preparation/20261116T234548Z_35065d09`.

| split | coppie | immagini distinte | file |
|---|---|---|---|
| train 20% | 6.000 | 4.989 | `05_subsets/20261116T234605Z_31f635c7/20/train_pairs.csv` |
| train 50% | 15.000 | 5.651 | `05_subsets/20261116T234605Z_31f635c7/50/train_pairs.csv` |
| train 100% | 30.000 | 5.681 | `05_subsets/20261116T234605Z_31f635c7/100/train_pairs.csv` |
| valid | 3.000 | 195 | `04_pairs/20261116T234604Z_cb0f352f/valid_pairs.csv` |
| test | 3.000 | 195 | `04_pairs/20261116T234604Z_cb0f352f/test_pairs.csv` |

Tutti gli split sono bilanciati (1/3 per esito). Prima del training il notebook verifica che:
- ogni esito coincida con la regola applicata ai semi delle cartelle;
- ogni coppia resti nel proprio split;
- i subset siano annidati (20 ⊂ 50 ⊂ 100);
- nessuna immagine sia condivisa tra train, valid e test.

## Modello

```
img1 ─► ResNet18 congelata (ImageNet, fc = Identity) ─► f1 (512) ┐
                                                                 ├─► concat (1024) ─► Linear 256 ─► ReLU ─► Dropout 0.2 ─► Linear 3
img2 ─► stessa ResNet18 (pesi condivisi)             ─► f2 (512) ┘                                     {p1_win, p2_win, draw}
```

- Backbone `resnet18(weights=IMAGENET1K_V1)`, 11,18 M parametri, **mai aggiornato**: `requires_grad=False` e sempre in
  `eval()` (BatchNorm fisse). Si allena solo la testa `DuelHead`, con **263.171 parametri**.
- Input: JPEG nativi 224×224 RGB, normalizzazione media/std ImageNet. Niente resize e niente `CenterCrop`, perché taglierebbe
  i bordi della carta. Niente augmentation.
- Supervisione: **solo l'esito** `(img1, img2, outcome)`, con `CrossEntropyLoss` sui 3 logit. I semi si usano solo nei controlli
  sui dati e nella diagnostica del §11.
- Ottimizzazione: con il backbone fisso le feature di ogni immagine non cambiano. Si calcolano quindi **una sola volta**
  (6.071 × 512) e la testa si allena su quelle. Il risultato è identico a quello che si avrebbe passando ogni volta le
  immagini nella rete, ma molto più veloce.

## Protocollo

| | |
|---|---|
| Ottimizzatore | Adam, lr 1e-3 |
| Batch | 32 |
| Epoche | max 30, early stopping con patience 5 sulla Macro-F1 di valid |
| Checkpoint | migliore Macro-F1 di valid, ricaricato prima del test |
| Seed | 0, 1, 2 per ciascuna percentuale (9 run in tutto) |
| Test | valutato una volta per run, sul checkpoint scelto con il valid; nessuna scelta fatta guardando il test |
| Tempi | estrazione delle feature, training della testa e inferenza misurati separatamente, con `cuda.synchronize` |

## Risultati

### Esito della partita (metrica principale)

| train | test accuracy | test Macro-F1 | valid Macro-F1 (checkpoint) | epoca migliore (media) | training testa | inferenza test (3.000 coppie) |
|---|---|---|---|---|---|---|
| 20% | 0.653 ± 0.017 | 0.653 ± 0.017 | 0.714 ± 0.019 | 18 | 8.0 s | 1.5 ms |
| 50% | 0.622 ± 0.043 | 0.622 ± 0.043 | 0.676 ± 0.061 | 13 | 15.0 s | 1.4 ms |
| 100% | 0.705 ± 0.009 | 0.705 ± 0.009 | 0.761 ± 0.023 | 9.7 | 27.2 s | 1.5 ms |

Media ± deviazione standard sui 3 seed. A questi tempi va aggiunta l'estrazione delle feature, fatta una volta sola:
14.9 s per 6.071 immagini (2.46 ms per immagine). Il tempo di inferenza riguarda solo la testa, con le feature già calcolate.

Dettaglio per seed:

| train | seed | epoca migliore / eseguite | valid F1 | test acc | test F1 |
|---|---|---|---|---|---|
| 20% | 0 | 17 / 22 | 0.736 | 0.668 | 0.668 |
| 20% | 1 | 13 / 18 | 0.707 | 0.634 | 0.634 |
| 20% | 2 | 24 / 29 | 0.701 | 0.657 | 0.657 |
| 50% | 0 | 5 / 10 | 0.611 | 0.574 | 0.574 |
| 50% | 1 | 19 / 24 | 0.733 | 0.654 | 0.655 |
| 50% | 2 | 15 / 20 | 0.684 | 0.638 | 0.638 |
| 100% | 0 | 9 / 14 | 0.763 | 0.708 | 0.707 |
| 100% | 1 | 11 / 16 | 0.783 | 0.713 | 0.713 |
| 100% | 2 | 9 / 14 | 0.737 | 0.695 | 0.695 |

### Dove sbaglia (test, seed 0, quota di coppie corrette per combinazione di semi)

| seme P1 | seme P2 | 20% | 50% | 100% |
|---|---|---|---|---|
| clubs | clubs | 0.676 | 0.598 | 0.652 |
| clubs | hearts | 0.706 | 0.619 | 0.790 |
| clubs | spades | 0.541 | 0.814 | 0.658 |
| hearts | clubs | 0.694 | 0.706 | 0.604 |
| hearts | hearts | 0.748 | 0.820 | 0.883 |
| hearts | spades | 0.811 | 0.787 | 0.787 |
| spades | clubs | 0.674 | **0.000** | 0.725 |
| spades | hearts | 0.596 | 0.814 | 0.692 |
| spades | spades | 0.572 | **0.009** | 0.581 |

### Diagnostica: le feature ImageNet separano già i semi?

È una regressione logistica lineare `feature 512 → seme`, allenata sulle immagini di train e valutata su quelle di test.
**Usa le etichette dei semi**, quindi non fa parte dell'Approccio B: serve solo a capire dove sta il limite.

- Accuratezza **0.826**, Macro-F1 0.826, su 195 immagini di test.
- Matrice di confusione (righe = vero, colonne = previsto):

| | hearts | clubs | spades |
|---|---|---|---|
| hearts | 57 | 2 | 6 |
| clubs | 3 | 52 | 10 |
| spades | 3 | 10 | 52 |

## Interpretazione

1. **Nessun effetto soffitto.** La Macro-F1 di test resta tra 0.62 e 0.71, lontana da 1. Con questa configurazione la ResNet18
   non rende il problema banale.
2. **Il limite è la percezione, non la regola.** Anche un classificatore lineare che vede le etichette dei semi si ferma a 0.83 e
   confonde soprattutto **fiori e picche** (entrambi neri, con forme simili). Le feature ImageNet congelate distinguono bene il
   rosso dal nero, ma non abbastanza la forma del seme. La testa non può recuperare un'informazione che le feature non contengono.
3. **La curva non è monotona (50% < 20%)** per colpa di un solo run. Con il seed 0 al 50% il training si ferma all'epoca 5
   (valid F1 0.611) e sbaglia quasi tutte le coppie spades-clubs (0.000 corrette) e spades-spades (0.009), mentre spades-hearts
   resta a 0.814.
   È uno stato degenere da cui la testa non esce prima che scatti la patience. Gli altri due seed al 50% (0.655 e 0.638) sono in
   linea con il 20%. La patience di 5 epoche è probabilmente troppo bassa per questo caso.
4. **Con più dati migliora poco**: da 0.65 al 20% a 0.71 al 100%, con 5 volte le coppie. Coerente con il punto 2: la testa ha già
   quasi tutto quello che le feature congelate possono dare.

## Confronto con la baseline di Ernesto (`feature/baseline`)

Ernesto ha implementato l'Approccio B con `BaselineDuelNet`: una CNN piccola stile Lab 03 allenata da zero, input 64×64,
74.851 parametri. In `docs/BASELINE_APPROACH_B.md` riporta solo risultati **preliminari sul validation**: non ha ancora
valutato il test, per scelta di protocollo.

| | ResNet18 congelata (questo notebook) | CNN da zero (`feature/baseline`) |
|---|---|---|
| Valid Macro-F1 20% | 0.714 ± 0.019 (3 seed; migliore 0.736) | **0.771** |
| Valid Macro-F1 50% | 0.676 ± 0.061 (3 seed; migliore 0.733) | **0.852** |
| Valid Macro-F1 100% | 0.761 ± 0.023 | non riportata |
| Test | riportato sopra | non ancora valutato |
| Seed | 0, 1, 2 | 42 (uno solo) |
| Input | 224×224, normalizzazione ImageNet | 64×64, normalizzazione [-1, 1] |
| Cosa si allena | solo la testa (263k parametri) | tutta la rete (75k parametri) |
| Training | max 30 epoche, early stopping (patience 5), batch 32 | 25 epoche, batch 64, lr 1e-3 |

Lettura:
- La CNN da zero è già **più forte** sul validation (+6 punti al 20%, +18 al 50%), pur essendo molto più piccola, perché impara
  feature specifiche per le carte. Lo conferma anche la sua curva a due gradini: prima separa rosso/nero, poi fiori/picche.
  È lo stesso confine su cui inciampano le feature congelate.
- Cautele: il suo dato viene da un solo seed (qui i seed al 50% vanno da 0.61 a 0.73) e da risoluzione, normalizzazione ed epoche
  diverse. Va anche verificato che abbia usato il run di coppie finale `20261116T234548Z_35065d09`, perché alcuni comandi nei
  suoi documenti citano ancora i CSV del 15/11.

## Limiti di questo esperimento

- Valid e test contengono solo **195 immagini** ciascuno, ripetute in 3.000 coppie: le coppie sono correlate e l'incertezza vera è
  più ampia di quanto suggerisca la deviazione standard sui seed.
- Tempi misurati su **Tesla T4 (Colab)**: non sono confrontabili con i run finali, previsti tutti sulla stessa macchina locale.
- Il test è stato calcolato a ogni run, pur senza scegliere nulla guardandolo. Il protocollo concordato con Ernesto prevede di
  valutarlo una volta sola alla fine: questi numeri di test vanno letti come esplorativi, non come risultato finale.
- Nessuna augmentation, nessun fine-tuning, una sola configurazione della testa (lr, hidden e dropout non ottimizzati).
- La cella §12 del notebook ("Come leggere i risultati") contiene criteri scritti prima del run. L'esito vero è in questo README.

## Cosa propongo per la scelta del modello

1. **Scartare la ResNet18 congelata** come backbone finale: perde contro una CNN piccola allenata da zero.
2. Confrontare sugli **stessi split, con 3 seed** e scegliendo solo sul validation:
   - (a) la CNN di Ernesto;
   - (b) la ResNet18 con **fine-tuning in due fasi**: prima solo la testa, poi `layer4` sbloccato con lr 1e-4 sul backbone.
   Il fine-tuning dovrebbe correggere proprio la confusione fiori/picche.
3. Per il fine-tuning alzare la patience (per esempio a 10) per evitare stop prematuri come quello del seed 0 al 50%.
4. Usare lo stesso backbone e lo stesso preprocessing sia per A sia per B, come previsto dal protocollo.

Giudicate voi i numeri e il ragionamento: se qualcosa non torna, scrivetelo e ne parliamo prima di fissare il backbone.
