# Hugging Face AI Studio — Streamlit

Ein KI-Chat mit Dashboard-Layout, Sidebar-Einstellungen und Hugging Face Inference API.

## Voraussetzungen

- Python 3.10 oder neuer
- Internetverbindung
- Hugging Face Access Token mit den erforderlichen Berechtigungen
- Modell, das über einen unterstützten Inference Provider für Chat-Completions verfügbar ist

## Installation

### 1. Projekt entpacken und Terminal öffnen

Wechsle im Terminal in den Projektordner.

### 2. Virtuelle Umgebung erstellen (empfohlen)

Windows:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Pakete installieren

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Hugging Face Token konfigurieren

Erstelle einen Token unter https://huggingface.co/settings/tokens. Gib den Token in der linken Seitenleiste der App ein, oder setze ihn als Umgebungsvariable.

Windows PowerShell:
```powershell
$env:HF_TOKEN="hf_dein_token"
```

macOS / Linux:
```bash
export HF_TOKEN="hf_dein_token"
```

Für eine Bereitstellung auf Streamlit Community Cloud kannst du den Token in den App-Secrets hinterlegen:
```toml
HF_TOKEN = "hf_dein_token"
```
In `app.py` kann `os.getenv("HF_TOKEN", "")` bei Bedarf um `st.secrets.get("HF_TOKEN", "")` ergänzt werden, damit Streamlit-Secrets ebenfalls automatisch gelesen werden.

**Sicherheit:** Niemals einen echten Token in `app.py` schreiben oder in ein öffentliches Git-Repository committen.

### 5. App starten

```bash
streamlit run app.py
```

Die lokale App ist normalerweise erreichbar unter:
http://localhost:8501

## Modell auswählen

Der Standardwert ist `Qwen/Qwen2.5-7B-Instruct`. Falls dieses Modell über deine aktuelle Inference-Provider-Konfiguration nicht verfügbar ist, wähle im Hugging Face Model Hub ein Chat-Modell mit passender Inference-Provider-Unterstützung und trage dessen Modell-ID in der Sidebar ein.

## Funktionen

- Dashboard mit Statuskennzahlen
- Chat-Verlauf in der aktuellen Sitzung
- Token-Streaming für Antworten
- Sidebar: Modell, Temperatur, Antwortlänge, Kontextfenster und Systemanweisung
- Schnellstart-Prompts
- Chat leeren und Verlauf als TXT exportieren
- verständliche Fehlermeldung mit aufklappbaren technischen Details

## Häufige Probleme

- **401/403:** Token fehlt, ist ungültig oder hat nicht die nötigen Berechtigungen.
- **404/Modell nicht verfügbar:** Modell-ID prüfen und kontrollieren, ob ein Inference Provider das Modell unterstützt.
- **429/Rate limit:** Später erneut versuchen oder den Provider-/Account-Limits folgen.
- **Keine Antwort:** Anderes unterstütztes Chat-Modell testen.
