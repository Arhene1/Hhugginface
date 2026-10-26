
# FundFinder – Digitales Fundbüro

Eine Streamlit-Web-App zur Verwaltung von Fundstücken mit
KI-gestützter Bilderkennung über Hugging Face.

## Funktionen
- Fotos hochladen und analysieren
- Gegenstände automatisch kategorisieren
- Fundstücke manuell erfassen und bearbeiten
- Suche und Filter nach Kategorie und Status
- CSV-Export
- Keine Datenbank

## Starten

```bash
pip install -r requirements.txt
streamlit run app.py
```

Für die KI-Fotoanalyse muss `HF_TOKEN` in den Streamlit Secrets
hinterlegt sein.
