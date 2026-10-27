# FundFinder – korrigierte Streamlit-Version

## Dateien
- `app.py`: Hauptanwendung
- `requirements.txt`: Abhängigkeiten

## Deployment auf Streamlit Community Cloud
1. Lade `app.py` und `requirements.txt` in dein GitHub-Repository hoch.
2. In Streamlit Community Cloud stelle als Main file path `app.py` ein.
3. Starte die App neu.

## Was diese Version verbessert
- Die Fundverwaltung funktioniert auch ohne Hugging-Face-Token.
- KI-Fehler wie 401, 403, 429 und Zeitüberschreitungen werden verständlich angezeigt.
- Bildanalyse ist optional; ein fehlgeschlagener KI-Aufruf verhindert nicht das manuelle Erfassen.
- Suche, Statusverwaltung und CSV-Export sind enthalten.

## Wichtige Grenzen
Diese Version nutzt `st.session_state` und damit keinen gemeinsamen Speicher. Andere Besucher sehen nicht automatisch dieselben Einträge; Daten können nach einem Neustart verloren gehen. Für gemeinsam sichtbare, dauerhaft gespeicherte Fundstücke ist ein gemeinsamer Speicher erforderlich (z. B. Datenbank oder ein anderer persistenter Dienst).

Der Hugging-Face-Token wird im Seitenmenü eingegeben und nur in der jeweiligen Sitzung verwendet. Für eine produktive App mit vielen Nutzern ist eine serverseitige Secret-Konfiguration sicherer. Ein Token darf niemals in den Quellcode oder in ein öffentliches GitHub-Repository geschrieben werden.
