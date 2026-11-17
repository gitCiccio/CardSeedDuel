# Organizzazione della validazione

`data/definitions.py` è la fonte unica per Enum, colonne/campi obbligatori,
ranghi, marker Joker, split, estensioni, campi dell'indice e ordine delle classi.
Sono valori immutabili: una classe astratta non aggiungerebbe un contratto utile.
Gli import precedenti da schema e validation_result restano compatibili.

`validation/validation_suite.py` coordina la pipeline. I componenti interni sono
in `validation/sub_validation/`:

- `validation_metadata.py`: header CSV, snapshot e conversione delle righe.
- `validation_paths.py`: riferimenti, percorsi pubblicabili e conflitti.
- `validation_classes.py`: coerenza degli indici e invarianti del risultato.
- `validation_images.py`: accettazione dei file e immagini non referenziate.
- `serialization.py`: righe serializzate, CSV, scrittura JSON e costruzione report.

Le funzioni condivise fra moduli hanno nomi pubblici all'interno di questo
package di implementazione; gli helper usati solo nel proprio file hanno `_`.
I chiamanti della pipeline usano validate_dataset ed export_validation.

Il logger del progetto sostituisce le chiamate logging.info globali. Gli INFO
descrivono le fasi e, su richiesta, il parsing e inserimento di ogni riga candidata. I problemi attesi diventano record di
scarto/review. logger.exception produce per default una riga compatta; usare
LoggingManager.configure(tracebacks=True) per il traceback completo in debug.
Questo formato riguarda i log, non elimina i traceback di eccezioni non gestite:
il punto di ingresso deve gestire gli errori attesi e restituire un codice di uscita.
I report condivisi usano messaggi controllati per non esportare percorsi locali.

Corrette due regressioni osservate prima del riordino: il controllo delle
colonne mancanti ora solleva solo quando necessario, e il parsing restituisce
rejected/candidates nello stesso ordine atteso dal coordinatore.

Non sono state eseguite pipeline sui dati originali durante il riordino. Test
funzionali e run definitivo saranno il passaggio successivo concordato.
