# Fundbüro KI – Streamlit

Eine kleine deutschsprachige Fundbüro-App mit SQLite und optionaler Hugging-Face-Inference-API.

## Funktionen
- Verlust- und Fundmeldungen erfassen
- Meldungen filtern und durchsuchen
- Status verwalten
- Verlust- und Fundmeldungen mit Hugging Face vergleichen
- Lokaler Stichwort-Fallback, falls kein HF-Token konfiguriert ist
- CSV-Export

## Starten

Python 3.10 oder neuer empfohlen.

```bash
python -m venv .venv
```

Windows:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Dann:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Hugging Face konfigurieren

1. Erstelle unter https://huggingface.co/settings/tokens einen API-Token mit den nötigen Berechtigungen.
2. Setze `HF_TOKEN` als Umgebungsvariable.

Windows PowerShell:
```powershell
$env:HF_TOKEN="hf_DEIN_TOKEN"
streamlit run app.py
```

macOS/Linux:
```bash
export HF_TOKEN="hf_DEIN_TOKEN"
streamlit run app.py
```

Optional kannst du ein anderes Inference-Modell setzen:
```bash
export HF_MODEL="Qwen/Qwen2.5-7B-Instruct"
```

Die Verfügbarkeit und Kosten von Inference-Modellen hängen von Hugging Face und deinem Konto ab. Wenn der API-Aufruf nicht klappt oder kein Token gesetzt ist, nutzt die App einen lokalen Stichwortvergleich.

## Hinweise für den echten Betrieb
- Diese Version ist ein Prototyp: keine Anmeldung, keine Rollen und keine Zugriffskontrolle.
- Nicht ungeschützt öffentlich ins Internet stellen.
- Kontaktangaben und Beschreibungen können personenbezogene Daten enthalten. Vor produktivem Einsatz Datenschutz, Einwilligung, Löschfristen und Zugriffsschutz klären.
- Keine geheimen Identifikationsmerkmale vollständig veröffentlichen; bei Rückgaben eine sichere Verifikation durchführen.
- Für produktiven Einsatz Authentifizierung, Rollen, Audit-Logs, Backups und sichere Kontaktvermittlung ergänzen.
