# Correzioni della review di F2.1/F2.2

## Coerenza dei metadati

`CardImage` confronta l'etichetta con valore e seme, rifiutando ad esempio
`ace of hearts` associato a Clubs. Joker usa la rappresentazione verificata
nel CSV originale: etichetta `joker`, rank `xxx`, seme assente.

## Package e dipendenze

`pyproject.toml` dichiara il layout src, Python >=3.11 e Pillow 13.0.0, versione
presente nell'ambiente verificato. `requirements.txt` contiene `-e .`: installa
il progetto editable e risolve le dipendenze dalla singola fonte pyproject.
Non è un lockfile completo dell'ambiente; dipendenze dei futuri modelli saranno
aggiunte quando implementate e verificate. Non occorre pytest per unittest.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/inspect_dataset.py
```

Eseguire i comandi nello stesso ambiente e dalla radice del checkout.
In alternativa: `card-inspect --project-root /percorso/del/checkout`.
I normali import funzionano dopo l'installazione senza PYTHONPATH manuale.

## Codice riutilizzabile

`scripts/inspect_dataset.py` è il wrapper di avvio, con radice predefinita ricavata
dal proprio percorso. Le funzioni sono in `card_suit_duel.data.inspection`:

| Funzione | Input espliciti / risultato |
|---|---|
| `validate_dataset_paths` | Radice dati e percorso CSV; controlla prerequisiti |
| `read_metadata` | Percorso CSV; restituisce righe |
| `summarize_metadata` | Righe; restituisce conteggi |
| `scan_dataset_files` | Radice dati; restituisce percorsi per split |
| `compare_metadata_and_files` | Righe, percorsi e radice dati; restituisce anomalie |
| `inspect_images` | Percorsi e radice dati; restituisce proprietà/leggibilità |
| `build_report` | Risultati e percorsi di contesto; compone il dizionario |
| `run_inspection` | Radice dati, CSV e radice progetto; orchestra e restituisce il report |
| `save_report` | Report e destinazione; scrive il JSON |
| `main` | Argomenti CLI; configura log, esegue, salva e restituisce il codice di uscita |

Le funzioni di analisi non configurano il logger e non scrivono il report.
`run_inspection` non modifica i dati né crea output. I percorsi per il report
devono essere contenuti nella radice progetto indicata. Solo il comando di
avvio configura file di log e salvataggio. Gli errori bloccanti producono un
codice di uscita non nullo; i file illeggibili sono registrati e la scansione continua.

## Validazione eseguita

- Test della coerenza etichetta/valore/seme e del marcatore Joker.
- Test esistenti del contratto e mapping.
- Test su dataset temporanei: file illeggibili, riferimenti mancanti, immagini
  non elencate, split assente, schema CSV errato e radici indipendenti.
- Installazione in ambiente isolato e import senza PYTHONPATH.
- Confronto del report reale con la versione precedente, mantenendo invariati
  conteggi, proprietà e anomalia `train/ace of clubs/output`.

Questi interventi risolvono la review; non implementano la pulizia di F2.3.

Esito della verifica del 12 novembre 2026: **28 test superati**. Installazione
editable riuscita in un ambiente virtuale separato con Python 3.12 e Pillow
13.0.0; import e test eseguiti in modalità isolata (`-I`) da una directory
temporanea, senza dipendere da PYTHONPATH o dalla configurazione di PyCharm.
Verificati anche gli help del wrapper e del comando installato `card-inspect`.

La riesecuzione di `run_inspection` sui dati reali ha prodotto un dizionario
esattamente uguale al report JSON precedente: 8.155 righe CSV, 8.154 immagini
leggibili e il solo riferimento senza immagine `train/ace of clubs/output`.
Il confronto non ha sovrascritto il report né modificato i dati raw.
