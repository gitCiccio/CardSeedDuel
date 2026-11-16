# Ambiente per F2.9 - Preprocessing e loader

## Configurazione verificata il 16 novembre 2026

- Python 3.12.14, ambiente locale `.venv`.
- Pillow 13.0.0.
- torch 2.15.0+cpu.
- torchvision 0.29.2+cpu.
- Progetto installato in modalità editable con extra `neural`.

`pyproject.toml` dichiara le versioni di torch e torchvision nell’extra
opzionale neural. I controlli dei dati basati solo su Pillow non obbligano
quindi a installare lo stack neurale. I test preprocessing vengono saltati
se torch/torchvision non sono installati; nell’ambiente verificato sono
stati eseguiti e superati.

## Ripristino effettuato

La vecchia .venv puntava a un Python base non disponibile. È stata
conservata in una cartella locale `.venv-backup-*`, esclusa da Git, e
ricreata nel percorso `.venv` usato dal progetto.

Per il ripristino è stato usato il Python 3.12.14 del runtime locale Codex.
Come ogni venv, l’ambiente dipende dalla disponibilità del suo Python base;
non è una cartella portabile da distribuire ai compagni. Se quel runtime
viene sostituito, ricreare la venv con un Python stabile disponibile sulla
macchina. Non versionare .venv o il suo backup.

## Ricreazione su un altro computer

Da un Python compatibile installato localmente, creare una propria venv
e usare il relativo interprete. Dalla radice del progetto:

```text
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -e ".[neural]"
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -B -m unittest discover -s tests -p "test_*.py"
```

In PowerShell anteporre `&` se il percorso dell’eseguibile è tra virgolette.
Questa configurazione usa CPU e non richiede CUDA. La variante GPU andrà
scelta in base all’hardware e alla configurazione degli esperimenti; non
è necessaria per verificare il preprocessing iniziale.

Le dipendenze base possono essere installate con `pip install -r requirements.txt`;
quel file mantiene `-e .` e non attiva automaticamente l’extra neural.

## Verifiche

193 test superati, inclusi nove nuovi test del preprocessing. Verificati
forma [C,H,W], float32, canali RGB, scala [0,1], resize esplicito,
determinismo, input invariato e validazione delle dimensioni.
`pip check` non segnala requisiti incompatibili. L’import del package
funziona anche con Python isolato, senza PYTHONPATH manuale.

Non sono stati installati DeepProbLog o altri pacchetti non necessari a
questo blocco. La scelta dell’architettura e della normalizzazione definitiva
rimane aperta.
