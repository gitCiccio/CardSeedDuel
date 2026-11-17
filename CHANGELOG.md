# Changelog e registro delle decisioni - F2.1–F2.10

Aggiornato il 17 novembre 2026. Questo documento ricostruisce le decisioni
della preparazione dati: **situazione, soluzione adottata e motivazione**.
Non significa che tutto il lavoro descritto sia nuovo nell’ultimo commit:
F2.1–F2.8 costituiscono la base già lavorata. Il perimetro del commit corrente
e il passaggio di consegne sono in [CHANGES.md](CHANGES.md).

## Stato complessivo

| Fase | Stato tecnico e risultato |
|---|---|
| F2.1 | Censimento verificato: 8155 righe CSV, 8154 immagini |
| F2.2 | Contratto CardImage e mapping condivisi implementati |
| F2.3 | 8154 riferimenti validi, una riga rifiutata |
| F2.4 | 6072 immagini dei semi del gioco |
| F2.5 | 6071 immagini dopo revisione di un duplicato con etichetta errata |
| F2.6 | Oracolo delle nove combinazioni verificato |
| F2.7 | 30000/3000/3000 coppie train/valid/test |
| F2.8 | Train annidati di 6000/15000/30000 coppie |
| F2.9 | Loader condiviso verificato integralmente nelle tre condizioni |
| F2.10 | Comando unico e riproducibilità locale verificati; criteri esterni pendenti |

Lo stato tecnico non sostituisce lo stato della board: F2.1–F2.8 risultavano
chiusi all’ultima lettura; gli aggiornamenti richiesti per F2.9/F2.10 sono
stati rifiutati dal connettore e non sono stati applicati.

## F2.1 - Esplorazione e censimento

### Situazione

Prima di pulire il dataset non erano noti con sufficiente precisione
conteggi reali, schema del CSV, proprietà delle immagini e corrispondenza
tra metadati e filesystem. Un conteggio delle sole righe non prova che
le immagini esistano e siano leggibili.

### Soluzione adottata

Script di ispezione in sola lettura, con funzioni riutilizzabili estratte
in `data/inspection.py`. Il report distingue CSV, filesystem, confronto
dei riferimenti e decodifica delle immagini. I percorsi nei report condivisi
sono relativi al progetto. Gli originali non vengono modificati.

### Perché

Serviva una base verificabile prima di prendere decisioni sui dati, utile
anche al rapporto finale. Separare i livelli di analisi impedisce di
interpretare righe del CSV come immagini necessariamente disponibili.

### Evidenze

- CSV: 8155 righe e 53 etichette; train 7625, valid 265, test 265.
- Filesystem: 8154 immagini; train 7624, valid 265, test 265.
- Immagini censite: JPEG, RGB, 224x224, tutte leggibili.
- Un riferimento senza immagine: `train/ace of clubs/output`.
- Nessuna immagine scansionata senza riferimento nel CSV.

## F2.2 - Schema dati e mapping

### Situazione

Nomi di colonne, semi, split ed esiti ripetuti nei moduli rischiavano di
divergere. L’indice originale delle 53 carte non coincide con l’indice
dei tre semi o dei tre esiti del duello.

### Soluzione adottata

Dataclass immutabile CardImage per image_id, percorso relativo, split,
etichetta originale, rango e seme. Enum e costanti in
`data/mapping/definitions.py`; conversioni e mapping read-only in
`data/mapping/mappings.py`. La dataclass verifica anche la coerenza
tra rango, etichetta e seme. Il Joker ha suit=None e rank=xxx.

| Indice | Dominio semi | Dominio esiti |
|---:|---|---|
| 0 | hearts | p1_win |
| 1 | clubs | p2_win |
| 2 | spades | draw |

Questi domini sono distinti: l’indice 0 non collega semanticamente cuori
a vittoria. I modelli devono usare le funzioni di mapping condivise.

### Perché

Il contratto unico evita etichette numeriche incompatibili tra i due
approcci. Gli errori vengono intercettati vicino alla sorgente. Non si
normalizzano silenziosamente stringhe o percorsi; esistenza dei file e
unicità nell’intera raccolta restano controlli separati.

Quadri e Joker sono rappresentabili nel raw: la loro esclusione appartiene
al filtro, non rende invalido il contratto originale.

## F2.3 - Integrità e coerenza

### Situazione

Il CSV conteneva un riferimento inesistente. Occorreva inoltre distinguere
errori di una riga, conflitti che richiedono revisione e problemi strutturali
che impediscono di analizzare tutto il dataset.

### Soluzione adottata

Separati parser della singola riga, controllo del file, risultati strutturati
e orchestrazione: `validation.py`, `validation_result.py`, `validation_suite.py`
e moduli `sub_validation`. Stati accepted, rejected e review; motivi
machine-readable; controllo dei percorsi anche dopo risoluzione dei link.

Riferimenti identici coerenti: conservare il primo valido. Conflitti di
percorso/classe/metadati: revisione. File mancanti o non decodificabili:
rifiuto della riga. Errori della struttura del dataset: interrompere.

### Perché

Non tutte le anomalie ammettono una correzione automatica. I risultati
strutturati permettono audit e fasi successive senza parsare il terminale.
Il CSV raw rimane una fonte immutata; gli indici derivati esprimono le scelte.

### Evidenze

Run `20261113T145404Z_ffac6207`: 8154 accepted, 1 rejected, 0 review,
0 immagini senza riferimento. Record CSV 122, class index 0,
`train/ace of clubs/output`: missing_file. Non manca tutta la classe
asso di fiori. Il CSV accettato ha 8154 record più l’intestazione.

## F2.4 - Filtro dei semi

### Situazione

Il raw comprende quattro semi e Joker; il gioco richiede soltanto cuori,
fiori e picche. Un’immagine fuori dal dominio può essere perfettamente valida.

### Soluzione adottata

Filtro sull’indice approvato usando SUIT_CLASSES. Conservate hearts,
clubs e spades; escluse diamonds e Joker con motivazione esplicita.
Nessuna cancellazione delle immagini o nuova suddivisione degli split.

### Perché

La selezione segue le regole del progetto, non una valutazione della qualità
dei file. Separare esclusioni di dominio e rifiuti di integrità rende i
conteggi interpretabili. Il rango rimane nei metadati ma non decide l’esito.

### Evidenze

Run `20261113T183831Z_d6db5632`: 6072 ammesse, 2082 escluse
(1957 quadri e 125 Joker). Le picche sono state conservate.

| Split | Prima | Ammesse | Escluse |
|---|---:|---:|---:|
| Train | 7624 | 5682 | 1942 |
| Valid | 265 | 195 | 70 |
| Test | 265 | 195 | 70 |

Semi ammessi: fiori 1936, cuori 1975, picche 2161. Non è stato effettuato
un bilanciamento delle immagini per seme.

## F2.5 - Duplicati e revisione manuale

### Situazione

Percorsi diversi possono riferirsi allo stesso contenuto; tra split diversi
questo può contaminare la valutazione. Copie identiche con etichette diverse
possono invece contraddire la supervisione anche all’interno del train.

### Soluzione adottata

Due impronte: SHA-256 dei byte del file e SHA-256 di modalità, dimensioni
e pixel decodificati, senza resize o conversioni implicite. Raggruppamento
per pixel identici. Metadati confrontati: original_label, rank, suit.

Per gruppi coerenti, conservare un rappresentante con priorità test, valid,
train e poi image_id lessicograficamente minore. Per conflitti di metadati,
revisionare l’intero gruppo senza scegliere un’etichetta automaticamente.
Gli scarti sono esclusioni dall’indice, mai cancellazioni dei file originali.

### Perché

Le copie nello stesso split alterano il peso degli esempi; quelle tra split
possono favorire una valutazione ingannevole. La priorità fissata prima
degli esperimenti preserva il riferimento di valutazione senza spostare
immagini. Una discordanza di etichette richiede un’evidenza visiva.

### Evidenze e risoluzione

Run automatico `20261114T173913Z_87fe4ccf`: un gruppo esatto, tutto train,
con `train/four of clubs/127.jpg` e `train/four of spades/120.jpg`.
Stessi byte e pixel, etichette discordanti: 6070 conservate, 2 in review,
0 scarti definitivi. Nessun gruppo esatto tra split.

La verifica visiva di entrambe mostra un quattro di picche. Nel run
`20261114T180134Z_manual_review` è conservato il riferimento spades ed
escluso quello clubs con motivo duplicate_with_incorrect_label.

Risultato: **6071 conservate, 1 esclusa, 0 review**. Split: 5681/195/195;
semi: 1935 fiori, 1975 cuori, 2161 picche. Il criterio di priorità degli
split non è servito a risolvere questo conflitto: è stata la revisione visiva.

In F2.10 la decisione viene resa riapplicabile tramite configurazione.
Il confronto percettivo dei quasi-duplicati non è stato implementato:
non si certifica l’assenza di ogni possibile leakage.

## F2.6 - Oracolo degli esiti

### Situazione

Per creare esempi supervisionati serviva un’etichetta corretta per ogni
duello, distinta dalla futura previsione del modello.

### Soluzione adottata

`game/oracle.py`: determine_outcome applica regole fisse ai semi reali.
Cuori batte fiori, fiori batte picche, picche batte cuori; semi uguali
pareggiano. Esito dalla prospettiva di P1. Nessun uso del rango, di reti
o di file immagine. Sono rifiutati semi esclusi e valori non conformi.

### Perché

Separare l’oracolo dal generatore permette di testare tutte le nove
combinazioni prima di produrre migliaia di etichette. Il modello deve
inferire l’esito dalle immagini: i semi reali non gli vengono forniti
come risposta già nota. Le regole ProbLog dovranno essere coerenti, ma
la loro verifica avverrà nell’integrazione F3A.

## F2.7 - Coppie ordinate riproducibili

### Situazione

L’indice descriveva carte singole. Occorrevano duelli condivisi dai due
approcci, senza mescolare gli split o enumerare milioni di combinazioni.

### Soluzione adottata

CardPair valida immagini distinte, stesso split e outcome dell’oracolo.
Coppie ordinate: (A,B) e (B,A) sono esempi diversi. Nessuna ripetizione
della stessa coppia ordinata; una carta può avere partner diversi.
pair_id è SHA-256 del JSON ordinato dei due image_id.

Per ogni split, gruppi di immagini ordinati per image_id. Capacità N x M
per semi diversi e N x (N-1) per lo stesso seme. Campionamento di indici
senza reinserimento; divmod decodifica le posizioni e salta l’auto-coppia.
Nel solo train esistono 32268080 combinazioni: non vengono materializzate.

Seed locali indipendenti 42/43/44. Quote uguali per esito e quasi uguali
tra le tre combinazioni di semi per esito. Resti assegnati con ordine
mescolato riproducibile. Se una quota supera la capacità, errore esplicito.

### Perché

L’ordine rappresenta i giocatori; l’isolamento degli split impedisce di
usare immagini train in valutazione. Campionare gli indici limita il costo
di memoria. Il benchmark bilanciato evita che gli esiti più frequenti
dominino il confronto; non pretende di riprodurre una distribuzione naturale.

### Evidenze

Run `20261115T105737Z_b3f5a3e3`: 30000 train (10000 per esito),
3000 valid e 3000 test (1000 per esito). Coppie tutte uniche, con tutte
le 6071 immagini partecipanti. I tre CSV pesano circa 4,85 MB: restano
leggibili e versionabili senza ZIP. I semi non sono colonne feature dei CSV.

## F2.8 - Train 20/50/100 annidati

### Situazione

Il confronto richiede quantità crescenti di training, preservando la
valutazione. Tre estrazioni indipendenti cambierebbero anche i membri
degli insiemi e renderebbero meno controllato il confronto.

### Soluzione adottata

`pairs_reader.py` ricostruisce le coppie e verifica riferimenti, ID, split
e outcome. `subset.py` raggruppa le coppie train per combinazione di semi,
ordina per pair_id e mescola ciascun gruppo una sola volta con seed 123.
Prefissi crescenti producono insiemi annidati, esiti bilanciati e quote
per combinazione quasi uguali. I resti seguono un ordine fisso tra percentuali.

La selezione è casuale ma l’ordine esportato segue il train sorgente. Il
100% conserva l’intero CSV; i ridotti non generano nuove coppie. Richiesta
divisibilità per 10 del conteggio per esito per percentuali esatte.

Valid e test rimangono quelli di F2.7: riferimenti relativi e fingerprint
nel manifest, senza copie in ogni cartella percentuale.

### Perché

20% incluso nel 50%, incluso nel 100% rende osservabile l’effetto dei dati
aggiunti. Le percentuali riguardano coppie, non immagini: una carta può
comparire in numerosi duelli. Non si riequilibrano quote impossibili in
modo silenzioso e non si usa test per scegliere gli esperimenti.

### Evidenze

Run `20261115T164052Z_742a2129`:

| Train | Coppie | Per esito | Immagini coinvolte |
|---|---:|---:|---:|
| 20% | 6000 | 2000 | 4989 |
| 50% | 15000 | 5000 | 5651 |
| 100% | 30000 | 10000 | 5681 |

Sottocartelle 20/50/100 con train_pairs.csv, manifest comune e log finale.
Validation e test hanno sempre 3000 coppie ciascuno.

## F2.9 - Immagini, tensori e batch

### Situazione

I CSV erano corretti ma non costituivano ancora input neurali. Le reti del
team non erano definite; occorreva evitare un caricamento accoppiato a
una particolare architettura e lasciare aperto il preprocessing finale.

### Soluzione adottata

- API pubbliche condivise in path_checks invece di importare helper
  privati o duplicare risoluzione, contenimento ed esistenza dei file.
- image_io restituisce una copia RGB in memoria con file sorgente chiuso.
- Preprocessing deterministico: dimensioni native, float32, [C,H,W] e
  scala [0,1]. Resize opzionale; niente augmentation o statistiche apprese
  dai dati. Nessuna normalizzazione specifica di una rete ancora da scegliere.
- CardPairDataset carica due immagini su richiesta; target long dal mapping
  condiviso. Gli ID restano diagnostici e non feature.
- DataLoader con seed locale, shuffle soltanto train, num_workers=0 nella
  prima versione Windows e drop_last=False per tutti gli split.
- Script check_pair_loaders legge il manifest F2.8 e verifica provenienza,
  conteggi, forme, tipi, finitezza, range e label dei batch osservati.

### Perché

Le parti comuni non richiedono conoscere la rete. L’iniezione della
trasformazione consente di sostituirla in seguito. Lettura lazy e file
chiusi limitano memoria e risorse; valutazione ordinata e completa evita
scarti impliciti. Il seed del loader controlla l’ordine, non rende
automaticamente riproducibili future augmentation casuali.

### Evidenze reali

Le tre prove iniziali avevano max-batches=1 ed erano parziali. Sono state
preservate e seguite da controlli completi:

| Train | Run completo | Train decodificato | Valid | Test |
|---|---|---:|---:|---:|
| 20% | 20261116T231929Z_f1190810 | 6000/6000 | 3000/3000 | 3000/3000 |
| 50% | 20261116T232034Z_d30b523f | 15000/15000 | 3000/3000 | 3000/3000 |
| 100% | 20261116T232221Z_e49c969d | 30000/30000 | 3000/3000 | 3000/3000 |

Tutti i flag all_pairs_decoded sono true; batch size 32, ultimi batch
train 16/24/16 e valid/test 24. Immagini float32 [B,3,224,224] in [0,1],
target int64 [B], esiti complessivi bilanciati. Le sorgenti fisse hanno
gli stessi hash in tutte le condizioni. Nessun modello addestrato.

Il criterio di #16 sui loader è verificato. L’adattatore Query/tensor source
è previsto dall’integrazione F3A (#3): non è dichiarato già implementato.

## F2.10 - Orchestrazione e riproducibilità

### Situazione

Le fasi erano avviabili separatamente e la revisione manuale del duplicato
non veniva riapplicata da un comando completo. Rieseguire gli script senza
quella decisione riportava il gruppo in review. Inoltre i report operativi
contengono timestamp, quindi non possono coincidere tra run diversi.

### Soluzione adottata

`scripts/prepare_dataset.py` usa `configs/preparation.json` per richiamare
ispezione, validazione, filtro, deduplicazione, coppie e sottoinsiemi tramite
le funzioni esistenti. Percorsi relativi, quantità e seed sono versionati.

La revisione visiva è in `configs/duplicate_review_decisions.json`: viene
applicata soltanto al gruppo con l’hash atteso e al rappresentante esplicito.
Conflitti nuovi, decisioni sconosciute, ripetute o incoerenti non vengono
risolti silenziosamente. Builder/export condivisi preservano il vecchio CLI.

Ogni run ha una nuova cartella e un summary finale. Gli output non possono
essere scritti nel raw. Le scritture non sono atomiche: una cartella parziale
non è un run concluso. I raw e i run precedenti restano invariati.

`reproducibility.json` confronta contenuti stabili: CSV originale, impronte
delle immagini del gioco, decisioni manuali versionate, quantità, seed e
hash dei nove indici. I report operativi mantengono run ID distinti.

### Perché

Un compagno deve poter ricostruire le decisioni senza conoscere la storia
della chat o modificare manualmente CSV. Separare identità dei dati e
identità dell’esecuzione evita un criterio impossibile di uguaglianza dei
timestamp e permette confronti verificabili dei contenuti.

### Evidenze e limiti di completamento

Due run completi: `20261116T232600Z_2dc1d17f` e
`20261116T232624Z_4d5f1014`. Il manifest semantico è identico byte per
byte. I nove indici coincidono anche con quelli approvati in precedenza.

```text
6880fe9dfd491615b199686b0f29318fa8eff4867dc1bd3835d69ad7766e872d
```

La verifica su clone di un compagno e il caricamento degli stessi ID/label
nei due modelli restano da attestare. Due esecuzioni locali non sostituiscono
questi criteri dell’issue #17. L’assenza di duplicati esatti non dimostra
l’assenza di fotografie quasi identiche.

## Decisioni trasversali e ambiente

- Immagini e CSV sono versionabili per scelta esplicita del gruppo; non
  si introducono ZIP obbligatori per piccoli indici testuali. Pesi .h5,
  checkpoint, credenziali, venv e backup restano esclusi.
- Recap in work_sessions, ora versionabile, e documenti tecnici in docs.
- Logger del progetto: colori per severità, livelli selezionabili, file
  senza ANSI e gestione leggibile delle eccezioni attese. I log finali delle
  nuove pipeline non ricostruiscono messaggi emessi prima dell’apertura del file.
- Package src installabile con pyproject ed editable install, evitando
  PYTHONPATH manuale. requirements.txt richiama il package base; l’extra
  neural abilita torch/torchvision.
- Venv ripristinata con backup. Configurazione CPU verificata: Python 3.12.14,
  Pillow 13.0.0, torch 2.15.0+cpu, torchvision 0.29.2+cpu. Non sono state
  scelte reti o pesi preaddestrati, né installato DeepProbLog.
- Il Python base della venv locale proviene dal runtime Codex: l’ambiente
  non è portabile. Ogni compagno deve creare la propria venv con un Python
  disponibile sulla propria macchina, seguendo la documentazione.

## Verifiche e consegna

**223 test superati** nella configurazione neurale completa. La suite
include test dei dati, regole, coppie, sottoinsiemi, accesso immagini,
preprocessing, loader, orchestrazione e replay della revisione.
Senza dipendenze neural i test neurali opzionali sono saltati: non equivale
a verificarli. I modelli e l’addestramento non sono coperti da questi risultati.

Riferimenti: [consegna del commit](CHANGES.md),
[riproducibilità](docs/F2_10_RIPRODUCIBILITA.md),
[evidenze loader](docs/F2_9_VERIFICA_COMPLETA.json),
[evidenze preparazione](docs/F2_10_VERIFICA_LOCALE.json).
