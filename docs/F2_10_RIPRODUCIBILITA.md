# F2.10 - Riprodurre la preparazione dati

## Comando principale

Dalla radice del repository, in un ambiente con il progetto installato:

```text
python scripts/prepare_dataset.py --config configs/preparation.json
```

Il comando legge il raw già estratto e versionato. Non scarica nuovamente
Kaggle, non cancella immagini e non sovrascrive i run precedenti.
Le funzioni richiamate sono nel package, non copiate nel comando CLI.

La configurazione contiene percorsi relativi al checkout, quantità per esito,
seed delle coppie 42/43/44 e seed sottoinsiemi 123. La decisione manuale
versionata ripristina in modo riproducibile la revisione visiva del quattro
di picche. Non è autorizzata una scelta automatica su conflitti nuovi.

## Output

Ogni run in `results/pipeline_output/preparation/<run_id>/` contiene:

```text
configuration.json
00_inspection.json
01_validation/
02_filtering/<run_id>/
03_deduplication/
04_pairs/<run_id>/
05_subsets/<run_id>/
reproducibility.json
summary.json
execution.log
```

`summary.json` indica i percorsi degli indici e il manifest dei sottoinsiemi,
relativi alla cartella del run principale. Il successivo checker dei loader
può usare quei percorsi al posto dei run storici di F2.7/F2.8.

Il comando registra un riepilogo finale nel log; i messaggi di avanzamento
appaiono nel terminale. Interrompe la preparazione se validazione o revisione
non consentono l’avanzamento. Una cartella parziale non è un risultato approvato.

## Quali contenuti devono essere identici

Run ID e timestamp sono diversi per evitare sovrascritture. Non è corretto
richiedere che i report operativi con quei valori coincidano byte per byte.
Il contratto riproducibile è invece:

- stesso contenuto di input, decisioni manuali, quantità e seed;
- stessi nove CSV prodotti;
- stesso `reproducibility.json` e relativo semantic_sha256.

Per una verifica registrata vedere `F2_10_VERIFICA_LOCALE.json`. Il controllo
locale ha confrontato due preparazioni complete e i CSV già approvati.

## Esecuzione dei test

```text
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -p "test_*.py"
```

Per test neurali e loader installare anche l’extra neural con la procedura
CPU documentata in F2_9_AMBIENTE_NEURALE.md. Senza quelle dipendenze i test
neurali opzionali sono saltati: non equivale a verificarli. Nell’ambiente
locale verificato sono stati eseguiti tutti i 223 test.

## Check di un compagno su clone pulito

1. Clonare il branch contenente codice, configurazioni, immagini e CSV.
2. Creare una propria venv con Python compatibile e installare il package.
3. Eseguire il comando principale con la configurazione versionata.
4. Confrontare reproducibility.json e semantic_sha256 con la verifica locale.
5. Registrare sistema, versioni e risultato in un recap o nell’issue #17.
6. Integrare i riferimenti nell’approccio DeepProbLog e nella baseline;
   verificare gli stessi ID e outcome, senza fornire i semi reali come input
   che sostituiscano il riconoscimento delle immagini.

Questa prova non è ancora stata effettuata da un compagno. I flag
companion_clone_verified e model_integration_verified nei run locali sono
quindi false. Non modificarli per dichiarazione senza un’evidenza registrata.

## Confine con F3A

Il loader PyTorch delle coppie è pronto e verificato su tutte le condizioni.
L’adattatore Dataset.to_query, tensor source e predicati neurali sono
previsti da F3A (#3), che deve mantenere classi e ID condivisi. Nessun modello
o training è già disponibile in questo checkout.

## Preparazione del commit

Versionare codice, configurazioni, test, documentazione e report scelti;
gli originali raw non cambiano. La .venv, i backup e i pesi rimangono locali.
Una regola gitignore non rimuove file già nello staging: controllare che
nessun .h5 sia incluso prima del commit.
