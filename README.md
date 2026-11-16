# Sonora — Mastering Desk

Applicazione desktop Windows in Python e CustomTkinter per un primo flusso di mastering locale.

## Funzioni

- Importa WAV, AIFF, FLAC e OGG; mostra forma d’onda e durata.
- Riproduce la traccia e permette di confrontare un riferimento.
- Catena con controlli di EQ, compressione, body, limiter, saturazione, warmth, clean, stereo width e ceiling.
- Preset Balanced, Warm, Loud e Reset; bypass e snapshot A/B/C.
- Stima RMS di loudness, picco, crest factor e correlazione stereo.
- Esporta il risultato in WAV PCM a 24 bit.
- L’audio viene elaborato sul computer; il meter LUFS è indicativo e non è un misuratore certificato EBU.

## Compilazione automatica con GitHub

Il workflow `.github/workflows/build-windows.yml` crea `Sonora.exe` su Windows quando fai push su `main`, pubblichi un tag `v*` o avvii il workflow manualmente.

1. Installa e apri GitHub Desktop, poi accedi al tuo account GitHub.
2. Scegli **File → Add local repository** e seleziona questa cartella del progetto.
3. Fai il commit iniziale e premi **Publish repository**; puoi scegliere un repository privato.
4. Su GitHub apri **Actions** e attendi il completamento di **Build Windows app**.
5. Apri l’esecuzione completata e scarica l’artefatto **Sonora-Windows**. Al suo interno trovi `Sonora.exe`.

La cartella è già inizializzata come repository Git locale. Il runner GitHub usa Windows e Python 3.12; non serve installare Python per eseguire l’eseguibile scaricato. La prima compilazione scarica i pacchetti da PyPI.

## Avvio locale per sviluppo

Su Windows con Python 3.12 installato, dalla cartella del progetto:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Per compilare a mano:

```powershell
python -m PyInstaller --noconfirm --clean --onefile --windowed --name Sonora --collect-all customtkinter --collect-all sounddevice --collect-all soundfile app.py
```

Il file generato sarà `dist\Sonora.exe`.

## Note

Sonora è un’app originale che riprende categorie di funzioni comuni nei software di mastering. Non include né riproduce il codice o gli algoritmi proprietari di Master Plan 2. La catena DSP è leggera e destinata ad ascolto e prototipazione: controlla il risultato a orecchio prima della pubblicazione.
