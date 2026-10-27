# FundFinder – überarbeitete Version

## Dateien
- `app.py`: Streamlit-App
- `requirements.txt`: Abhängigkeiten

## Lokal starten
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Hugging Face (optional)
Die App funktioniert auch ohne KI-Zugang. Wenn du die Bilderkennung aktivieren möchtest, setze `HF_TOKEN` als Umgebungsvariable oder als Streamlit Secret.

In Streamlit Community Cloud:
1. Öffne dein App-Dashboard.
2. Öffne **Settings → Secrets**.
3. Trage ein:
   ```toml
   HF_TOKEN = "dein_token"
   ```
4. Speichere und starte die App neu.

Der Token darf **nicht** in den Python-Code oder in ein öffentliches GitHub-Repository geschrieben werden. Ohne Token bleibt die manuelle Erfassung aktiv.

## In Streamlit Community Cloud bereitstellen
1. Lade `app.py` und `requirements.txt` in dein Repository hoch.
2. Wähle beim Erstellen/Ändern der App `app.py` als Main file path.
3. Falls du den bisherigen Dateinamen `Hugginface 3.0.py` beibehalten möchtest, musst du entweder den Main file path darauf setzen oder die Datei in `app.py` umbenennen.

## Wichtiger Speicherhinweis
Diese Version nutzt eine JSON-Datei und keine Datenbank. Das erlaubt einfache gemeinsame Nutzung während die App-Instanz läuft, ist aber **kein verlässlicher dauerhafter Speicher auf Streamlit Community Cloud**. Daten und hochgeladene Bilder können bei Neustarts, Rebuilds oder Änderungen an der Instanz verloren gehen. Für dauerhaft gemeinsame Daten mehrerer Nutzer brauchst du persistenten Speicher (z. B. einen externen Speicher oder eine Datenbank).

## Datenschutz
Fundstücke und Bilder können persönliche Informationen enthalten. Lade nur Inhalte hoch, die für den Fundprozess notwendig sind. Diese App hat keine Benutzeranmeldung oder Rollenverwaltung.
