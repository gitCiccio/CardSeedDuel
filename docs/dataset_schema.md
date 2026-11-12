# Contratto dei metadati e mapping

`CardImage` in `card_suit_duel.data.schema` contiene metadati immutabili, non pixel.
Il costruttore rifiuta valori non conformi con `MetadataError`; non legge il disco.

| Campo | Contratto |
|---|---|
| `image_id` | Stringa non vuota, uguale a `path` nello schema attuale |
| `path` | Percorso relativo normalizzato a `data/raw/`, con `/`, senza `..` |
| `split` | Enum `DatasetSplit`: train, valid, test; coincide con la prima parte del percorso |
| `original_label` | Etichetta originale; per carte normali deve essere `rank + " of " + suit.value` |
| `rank` | Valore originale; per Joker è il marcatore `xxx` del CSV |
| `suit` | Enum `CardSuit`; `None` esclusivamente per Joker |

I campi testuali non ammettono spazi iniziali/finali. Il percorso contiene almeno
split, cartella della classe e nome del file. L'esistenza del file, la coerenza della
cartella con l'etichetta e l'unicità degli identificativi sono controlli della
preparazione del dataset, non di una singola dataclass.

```python
from card_suit_duel.data.schema import CardImage, CardSuit, DatasetSplit

card = CardImage(
    image_id="train/ace of hearts/001.jpg",
    path="train/ace of hearts/001.jpg",
    split=DatasetSplit.TRAIN,
    original_label="ace of hearts",
    rank="ace",
    suit=CardSuit.HEARTS,
)
```

Quadri è rappresentabile nel dataset originale; viene escluso dal dominio del
duello. Joker ha etichetta `joker`, rank `xxx` e seme `None`.

## Ordine condiviso delle classi

| Indice | Seme | Esito |
|---:|---|---|
| 0 | hearts | p1_win |
| 1 | clubs | p2_win |
| 2 | spades | draw |

Semi ed esiti sono domini distinti: l'indice non associa un seme a un esito.
`SUIT_CLASSES` e `OUTCOME_CLASSES` in `mappings.py` definiscono l'ordine; mapping
read-only e nomi simbolici derivano da queste tuple. Non duplicare l'ordine nei
modelli. `suit_to_index`, `index_to_suit`, `outcome_to_index` e `index_to_outcome`
sono le conversioni comuni. Etichette sconosciute e semi esclusi causano
`InvalidMappingValueError`; indici negativi, fuori intervallo o di tipo errato
causano `InvalidMappingIndexError`. Nessuna pulizia implicita delle stringhe.

La conversione delle righe reali del CSV in questi oggetti resta prevista in F2.3.
