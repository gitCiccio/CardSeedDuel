# F2.9 - Scaletta e contratto di ingresso

Data: 15 novembre 2026. Stato: pianificazione e primo contratto; nessun
loader o preprocessing neurale implementato in questa sessione.

## Chiarimento sulla dipendenza dai modelli

Il gruppo non ha ancora scelto le architetture o l’uso di pesi preaddestrati.
Questo non blocca tutto F2.9: sono implementabili le parti comuni e i punti
di estensione. Il task completo richiederà poi una verifica con i modelli.
Non si impone una rete per poter procedere con la preparazione dei dati.

## Scaletta operativa

| Passo | Lavoro | Criterio di completamento |
|---|---|---|
| 1 | Contratto di ingresso e decisioni aperte | Metadati, classi e interfacce descritti; proposte distinte dalle decisioni |
| 2 | Lettura sicura delle immagini | Apertura e chiusura corrette, contenimento dei percorsi, errori leggibili |
| 3 | Preprocessing configurabile | Trasformazione passata dall’esterno; percorso deterministico iniziale verificato |
| 4 | Dataset delle coppie | Lettura di un esempio con due immagini, etichetta e riferimenti diagnostici |
| 5 | DataLoader | Batch corretti, seed controllato, valutazione completa e riproducibile |
| 6 | Selezione degli esperimenti | Lettura manifest F2.8, train 20/50/100 e valid/test fissi |
| 7 | Adattamento DeepProbLog | Riferimenti immagini e query compatibili con il modello del gruppo |
| 8 | Test e ispezione visiva | Batch sintetico e reale verificati, immagini prima/dopo leggibili |
| 9 | Documentazione e integrazione | Esempio d’uso dei due approcci e decisioni definitive registrate |

Passi 2, 4 e 6 non richiedono conoscere l’architettura. Il passo 3 può
definire l’interfaccia e un preprocessing iniziale; la trasformazione finale
va allineata con la rete. Il passo 7 dipende dal programma DeepProbLog.

## Punto 1 - Evidenze dal repository

Nel checkout ispezionato, models/ e logic/ contengono file descrittivi,
train.py non contiene un’implementazione e configs/ contiene la descrizione
del suo scopo. Non risultano implementate reti o trasformazioni.
pyproject.toml dichiara Pillow; PyTorch, torchvision e DeepProbLog non sono
ancora dipendenze dichiarate del pacchetto.

Le descrizioni dei file Keras nel README non costituiscono una scelta di
architettura: il README specifica che quei file non sono necessari e che
si prevedono modelli PyTorch dedicati.

## Contratto già disponibile

- Originali sotto data/raw; accesso tramite indice approvato e CardImage.
- Coppie ordinate, immagini distinte nello stesso split, esiti dell’oracolo.
- Train selezionabile tramite manifest F2.8, senza nuovi campionamenti.
- Validation e test fissi provenienti da F2.7.
- Ordine classi da mapping.definitions e mapping.mappings, senza riscriverlo.

| Mapping | Ordine |
|---|---|
| Semi | hearts=0, clubs=1, spades=2 |
| Esiti | p1_win=0, p2_win=1, draw=2 |

Il loader dovrà chiamare outcome_to_index e non duplicare le corrispondenze.
I semi reali sono metadati per controlli; non entrano come soluzione già
nota negli input del classificatore dei duelli.

## Interfacce proposte, indipendenti dall’architettura

1. Una funzione legge la CardImage e restituisce un’immagine in memoria,
   indipendente dal file già chiuso. Non ridimensiona né normalizza.
2. Il Dataset riceve una trasformazione configurabile, applicata a entrambe
   le immagini. La trasformazione specifica della rete non è codificata
   rigidamente nel lettore.
3. Per la baseline, contratto proposto: due tensori separati e un indice
   di esito; identificativi disponibili per diagnosi, separati dalle feature.
4. DeepProbLog riutilizza lettura e trasformazione; un adattatore separato
   gestirà i riferimenti neurali e le query secondo il modello implementato.

Non si decide ora se la baseline concatenerà immagini, feature o uscite:
questa è una responsabilità del modello.

## Preprocessing iniziale proposto, non ancora approvato né implementato

Per verificare il caricamento senza pesi preaddestrati, una base possibile
è mantenere RGB e dimensioni native 224x224, convertire in float32 con
ordine canali-altezza-larghezza e scalare i pixel in [0,1]. Il tensore per
immagine avrebbe forma [3,224,224]; il batch [B,3,224,224].

Sono proposte per un test iniziale, non requisiti definitivi dei modelli.
Un’architettura preaddestrata potrebbe richiedere dimensioni e trasformazioni
diverse: saranno configurate senza riscrivere la lettura degli indici.

All’inizio niente augmentation casuale o statistiche stimate dal dataset.
L’eventuale normalizzazione tramite medie/deviazioni dei dati richiederà una
politica esplicita: usare immagini del training completo anche nel caso 20%
fornisce informazione non supervisionata aggiuntiva e va dichiarato. Non
si useranno mai valid/test per stimare le statistiche del preprocessing.

## Decisioni da concordare prima dell’integrazione finale

- Architettura dei due approcci e dimensione effettiva di ingresso.
- Pesi preaddestrati e relativo preprocessing, se previsti.
- Supervisione soltanto sugli esiti o anche sui semi: un confronto deve
  rendere esplicite eventuali informazioni aggiuntive.
- Interfaccia delle query e dei tensori della versione DeepProbLog usata.
- Batch size e configurazione dei worker per l’ambiente del gruppo.

## Prossima attività eseguibile

Procedere al passo 2: estrarre e riutilizzare il controllo pubblico dei
percorsi e implementare la lettura di un’immagine con risorse correttamente
chiuse. Questa attività non richiede PyTorch né una rete già scelta.
La scaffalatura delle trasformazioni e dei Dataset seguirà mantenendo
configurabili le decisioni ancora aperte.
