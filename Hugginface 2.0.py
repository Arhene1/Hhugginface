
import os
import io
import json
import base64
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from openai import OpenAI


# --------------------------------------------------
# KONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FundFinder | Digitales Fundbüro",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

KATEGORIEN = [
    "Schlüssel",
    "Tasche / Rucksack",
    "Geldbörse",
    "Kleidung",
    "Elektronik",
    "Dokumente / Ausweise",
    "Schmuck / Uhr",
    "Brille",
    "Schirm",
    "Flasche / Trinkgefäß",
    "Spielzeug",
    "Sonstiges",
]

STATUS_OPTIONEN = [
    "Abgegeben",
    "Wird gesucht",
    "Zurückgegeben",
]

# --------------------------------------------------
# STIL
# --------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }
    .hero {
        padding: 1.6rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #142b4a, #236b78);
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        color: white;
        margin-bottom: .35rem;
    }
    .hero p {
        color: #e4f3f5;
        margin-bottom: 0;
    }
    div[data-testid="stMetric"] {
        padding: 1rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SITZUNGSSPEICHER – KEINE DATENBANK
# --------------------------------------------------

if "fundstuecke" not in st.session_state:
    st.session_state.fundstuecke = []

if "ki_ergebnis" not in st.session_state:
    st.session_state.ki_ergebnis = None

if "letzter_upload" not in st.session_state:
    st.session_state.letzter_upload = None


# --------------------------------------------------
# HUGGING FACE-ZUGANG
# --------------------------------------------------

def hole_hf_token():
    """Liest das Token aus Streamlit Secrets oder einer Umgebungsvariable."""
    token = os.getenv("HF_TOKEN")

    if token:
        return token

    try:
        return st.secrets["HF_TOKEN"]
    except (KeyError, FileNotFoundError):
        return None


def hole_modell():
    """Das Modell kann über Secrets oder Umgebungsvariablen geändert werden."""
    model = os.getenv("HF_IMAGE_MODEL")

    if model:
        return model

    try:
        return st.secrets["HF_IMAGE_MODEL"]
    except (KeyError, FileNotFoundError):
        return "Qwen/Qwen3-VL-30B-A3B-Instruct"


def bereite_bild_vor(upload):
    """Dreht das Bild korrekt, verkleinert es und erzeugt JPEG-Bytes."""
    bild = Image.open(upload)
    bild = ImageOps.exif_transpose(bild).convert("RGB")
    bild.thumbnail((1280, 1280))

    puffer = io.BytesIO()
    bild.save(puffer, format="JPEG", quality=85, optimize=True)

    return puffer.getvalue()


def analysiere_foto(bild_bytes):
    """
    Sendet das Foto an ein visionfähiges Modell über die
    OpenAI-kompatible Hugging Face Inference API.
    """
    token = hole_hf_token()

    if not token:
        raise ValueError(
            "Das Hugging-Face-Token fehlt. "
            "Bitte HF_TOKEN in Streamlit Secrets eintragen."
        )

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )

    bild_base64 = base64.b64encode(bild_bytes).decode("utf-8")
    bild_url = f"data:image/jpeg;base64,{bild_base64}"

    prompt = f"""
    Du bist eine KI für ein digitales Fundbüro.
    Analysiere das hochgeladene Foto eines möglicherweise
    verlorenen Gegenstands.

    Erfinde keine Details, die auf dem Bild nicht erkennbar sind.
    Nenne keine persönlichen Identitäten. Falls Dokumente oder
    Ausweise zu sehen sind, beschreibe sie nur allgemein und gib
    keine darauf lesbaren persönlichen Daten wieder.

    Wähle genau eine Kategorie aus dieser Liste:
    {", ".join(KATEGORIEN)}

    Antworte ausschließlich mit einem gültigen JSON-Objekt
    in diesem Format:
    {{
      "gegenstand": "kurzer Name des Gegenstands",
      "kategorie": "eine Kategorie aus der Liste",
      "beschreibung": "neutrale Beschreibung sichtbarer Merkmale",
      "sicherheit": 0.0
    }}

    "sicherheit" ist eine Schätzung zwischen 0 und 1.
    Wenn du unsicher bist, wähle "Sonstiges" und einen niedrigen Wert.
    """

    antwort = client.chat.completions.create(
        model=hole_modell(),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": bild_url},
                    },
                ],
            }
        ],
        max_tokens=500,
        temperature=0.1,
    )

    inhalt = antwort.choices[0].message.content

    if not inhalt:
        raise ValueError("Das KI-Modell hat keine Antwort geliefert.")

    # Entfernt mögliche Markdown-Codeblöcke um das JSON.
    inhalt = inhalt.strip()
    if inhalt.startswith("```"):
        zeilen = inhalt.splitlines()
        if zeilen:
            zeilen = zeilen[1:]
        if zeilen and zeilen[-1].strip().startswith("```"):
            zeilen = zeilen[:-1]
        inhalt = "\n".join(zeilen).strip()

    ergebnis = json.loads(inhalt)

    kategorie = ergebnis.get("kategorie", "Sonstiges")
    if kategorie not in KATEGORIEN:
        kategorie = "Sonstiges"

    try:
        sicherheit = float(ergebnis.get("sicherheit", 0))
    except (TypeError, ValueError):
        sicherheit = 0.0

    sicherheit = max(0.0, min(1.0, sicherheit))

    return {
        "gegenstand": str(ergebnis.get("gegenstand", "Unbekannter Gegenstand")),
        "kategorie": kategorie,
        "beschreibung": str(
            ergebnis.get("beschreibung", "Keine Beschreibung verfügbar.")
        ),
        "sicherheit": sicherheit,
    }


# --------------------------------------------------
# KOPFBEREICH
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🔎 FundFinder</h1>
        <p>Das intelligente digitale Fundbüro – Gegenstände erkennen,
        Fundstücke erfassen und schneller wiederfinden.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("📍 Navigation")
    st.caption("Digitales Fundbüro")

    seite = st.radio(
        "Bereich auswählen",
        [
            "Dashboard",
            "Fundstück erfassen",
            "Fundstücke durchsuchen",
            "Daten exportieren",
        ],
    )

    st.divider()
    st.caption("🤖 Bilderkennung mit Hugging Face")
    st.caption("💾 Speicherung nur in der aktuellen Sitzung")
    st.caption("🔒 Keine Datenbank, keine dauerhafte Speicherung")


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if seite == "Dashboard":
    fundstuecke = st.session_state.fundstuecke

    gesamt = len(fundstuecke)
    abgegeben = sum(
        f["status"] == "Abgegeben" for f in fundstuecke
    )
    gesucht = sum(
        f["status"] == "Wird gesucht" for f in fundstuecke
    )
    zurueckgegeben = sum(
        f["status"] == "Zurückgegeben" for f in fundstuecke
    )

    st.subheader("Dashboard")
    st.write("Übersicht über die Fundstücke in dieser Sitzung.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alle Fundstücke", gesamt)
    c2.metric("Abgegeben", abgegeben)
    c3.metric("Wird gesucht", gesucht)
    c4.metric("Zurückgegeben", zurueckgegeben)

    st.divider()
    st.subheader("Zuletzt erfasste Fundstücke")

    if not fundstuecke:
        st.info(
            "Noch keine Fundstücke vorhanden. "
            "Öffne „Fundstück erfassen“, um ein Foto hochzuladen."
        )
        if st.button("＋ Erstes Fundstück erfassen", type="primary"):
            st.session_state["navigation_ziel"] = "Fundstück erfassen"
            st.rerun()
    else:
        for fund in reversed(fundstuecke[-5:]):
            with st.container(border=True):
                links, rechts = st.columns([1, 3])

                with links:
                    st.image(fund["foto"], use_container_width=True)

                with rechts:
                    st.markdown(f"**{fund['titel']}**")
                    st.write(f"**Kategorie:** {fund['kategorie']}")
                    st.write(f"**Status:** {fund['status']}")
                    st.caption(
                        f"Erfasst am {fund['datum']} · ID: {fund['id'][:8]}"
                    )
                    st.write(fund["beschreibung"])


# --------------------------------------------------
# FUNDSTÜCK ERFASSEN
# --------------------------------------------------

elif seite == "Fundstück erfassen":
    st.subheader("＋ Neues Fundstück erfassen")
    st.write(
        "Lade ein Foto hoch. Die KI erstellt einen Vorschlag, "
        "den du vor dem Speichern überprüfen und bearbeiten kannst."
    )

    upload = st.file_uploader(
        "Foto des Gegenstands",
        type=["jpg", "jpeg", "png", "webp"],
        help="Unterstützte Formate: JPG, PNG und WEBP.",
        key="foto_upload",
    )

    bild_bytes = None

    if upload is not None:
        try:
            bild_bytes = bereite_bild_vor(upload)
            st.image(
                bild_bytes,
                caption="Hochgeladenes Foto",
                width=420,
            )
        except Exception as exc:
            st.error(f"Das Bild konnte nicht gelesen werden: {exc}")

    if upload is None:
        st.session_state.ki_ergebnis = None
        st.session_state.letzter_upload = None

    if bild_bytes is not None:
        # Erkennt einen neuen Upload, damit alte Ergebnisse nicht
        # versehentlich für ein anderes Bild übernommen werden.
        if st.session_state.letzter_upload != bild_bytes:
            st.session_state.ki_ergebnis = None
            st.session_state.letzter_upload = bild_bytes

        if st.button(
            "🤖 Foto mit Hugging Face analysieren",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Die KI analysiert den Gegenstand ..."):
                try:
                    st.session_state.ki_ergebnis = analysiere_foto(bild_bytes)
                    ergebnis = st.session_state.ki_ergebnis

                    # Vorbelegung der bearbeitbaren Eingabefelder.
                    st.session_state["eingabe_titel"] = ergebnis["gegenstand"]
                    st.session_state["eingabe_kategorie"] = ergebnis["kategorie"]
                    st.session_state["eingabe_beschreibung"] = (
                        ergebnis["beschreibung"]
                    )

                    st.success("Fotoanalyse abgeschlossen.")
                except Exception as exc:
                    st.error(
                        "Die Fotoanalyse ist fehlgeschlagen. "
                        "Prüfe dein Hugging-Face-Token, den Modellnamen "
                        "und die Verfügbarkeit des Modells."
                    )
                    st.code(str(exc))

    ergebnis = st.session_state.ki_ergebnis

    if ergebnis:
        st.success(
            f"KI-Vorschlag: {ergebnis['gegenstand']} "
            f"· geschätzte Sicherheit: {ergebnis['sicherheit']:.0%}"
        )

    st.divider()
    st.markdown("#### Angaben zum Fundstück")

    # Die Felder können auch ohne KI manuell ausgefüllt werden.
    with st.form("fundstueck_formular"):
        titel = st.text_input(
            "Name des Gegenstands *",
            key="eingabe_titel",
            placeholder="z. B. Schwarzer Rucksack",
        )

        kategorie = st.selectbox(
            "Kategorie",
            KATEGORIEN,
            key="eingabe_kategorie",
        )

        beschreibung = st.text_area(
            "Beschreibung",
            key="eingabe_beschreibung",
            placeholder="Farbe, Marke, besondere Merkmale ...",
        )

        ort = st.text_input(
            "Fundort",
            placeholder="z. B. Bibliothek, Raum 204",
        )

        datum_fund = st.date_input(
            "Funddatum",
            value=datetime.now().date(),
        )

        status = st.selectbox(
            "Status",
            STATUS_OPTIONEN,
            index=0,
        )

        speichern = st.form_submit_button(
            "Fundstück speichern",
            type="primary",
            use_container_width=True,
        )

    if speichern:
        if bild_bytes is None:
            st.error("Bitte lade zuerst ein Foto hoch.")
        elif not titel.strip():
            st.error("Bitte gib einen Namen für den Gegenstand ein.")
        else:
            neues_fundstueck = {
                "id": uuid.uuid4().hex,
                "titel": titel.strip(),
                "kategorie": kategorie,
                "beschreibung": beschreibung.strip(),
                "ort": ort.strip(),
                "funddatum": datum_fund.isoformat(),
                "status": status,
                "datum": datetime.now().strftime("%d.%m.%Y %H:%M"),
                "foto": bild_bytes,
                "ki_sicherheit": (
                    ergebnis["sicherheit"] if ergebnis else None
                ),
                "ki_modell": hole_modell() if ergebnis else None,
            }

            st.session_state.fundstuecke.append(neues_fundstueck)
            st.session_state.ki_ergebnis = None
            st.session_state.letzter_upload = None

            st.success("Fundstück wurde in dieser Sitzung gespeichert.")
            st.info(
                "Hinweis: Die Daten werden nicht dauerhaft gespeichert. "
                "Bei einem Neustart oder Sitzungsverlust können sie verloren gehen."
            )


# --------------------------------------------------
# FUNDSTÜCKE SUCHEN UND VERWALTEN
# --------------------------------------------------

elif seite == "Fundstücke durchsuchen":
    st.subheader("🔍 Fundstücke durchsuchen")

    fundstuecke = st.session_state.fundstuecke

    if not fundstuecke:
        st.info("Es sind noch keine Fundstücke vorhanden.")
    else:
        suchtext = st.text_input(
            "Suche",
            placeholder="Name, Beschreibung oder Fundort eingeben ...",
        )

        f1, f2 = st.columns(2)

        with f1:
            filter_kategorie = st.selectbox(
                "Kategorie filtern",
                ["Alle Kategorien"] + KATEGORIEN,
            )

        with f2:
            filter_status = st.selectbox(
                "Status filtern",
                ["Alle Status"] + STATUS_OPTIONEN,
            )

        gefiltert = []

        for fund in fundstuecke:
            suchfelder = " ".join(
                [
                    fund["titel"],
                    fund["beschreibung"],
                    fund["ort"],
                    fund["kategorie"],
                ]
            ).casefold()

            if suchtext.casefold() not in suchfelder:
                continue

            if (
                filter_kategorie != "Alle Kategorien"
                and fund["kategorie"] != filter_kategorie
            ):
                continue

            if (
                filter_status != "Alle Status"
                and fund["status"] != filter_status
            ):
                continue

            gefiltert.append(fund)

        st.caption(f"{len(gefiltert)} Fundstück(e) gefunden")

        for fund in reversed(gefiltert):
            with st.container(border=True):
                bild_spalte, info_spalte = st.columns([1, 2])

                with bild_spalte:
                    st.image(fund["foto"], use_container_width=True)

                with info_spalte:
                    st.markdown(f"### {fund['titel']}")
                    st.write(f"**Kategorie:** {fund['kategorie']}")
                    st.write(f"**Fundort:** {fund['ort'] or 'Nicht angegeben'}")
                    st.write(f"**Funddatum:** {fund['funddatum']}")
                    st.write(f"**Beschreibung:** {fund['beschreibung']}")
                    st.caption(
                        f"Erfasst: {fund['datum']} · ID: {fund['id'][:8]}"
                    )

                    status_key = f"status_{fund['id']}"
                    if status_key not in st.session_state:
                        st.session_state[status_key] = fund["status"]

                    neuer_status = st.selectbox(
                        "Status ändern",
                        STATUS_OPTIONEN,
                        key=status_key,
                    )

                    if neuer_status != fund["status"]:
                        fund["status"] = neuer_status
                        st.success("Status aktualisiert.")

                    if st.button(
                        "🗑️ Fundstück löschen",
                        key=f"loeschen_{fund['id']}",
                    ):
                        st.session_state.fundstuecke = [
                            f for f in st.session_state.fundstuecke
                            if f["id"] != fund["id"]
                        ]
                        st.session_state.pop(status_key, None)
                        st.rerun()


# --------------------------------------------------
# CSV-EXPORT
# --------------------------------------------------

elif seite == "Daten exportieren":
    st.subheader("⬇️ Fundstücke exportieren")
    st.write(
        "Exportiere die erfassten Angaben als CSV-Datei. "
        "Die Fotos werden nicht in die CSV-Datei eingebettet."
    )

    fundstuecke = st.session_state.fundstuecke

    if not fundstuecke:
        st.info("Es gibt noch keine Daten zum Exportieren.")
    else:
        export_daten = []

        for fund in fundstuecke:
            export_daten.append(
                {
                    "ID": fund["id"],
                    "Gegenstand": fund["titel"],
                    "Kategorie": fund["kategorie"],
                    "Beschreibung": fund["beschreibung"],
                    "Fundort": fund["ort"],
                    "Funddatum": fund["funddatum"],
                    "Status": fund["status"],
                    "Erfasst am": fund["datum"],
                    "KI-Sicherheit": fund["ki_sicherheit"],
                    "KI-Modell": fund["ki_modell"],
                }
            )

        df = pd.DataFrame(export_daten)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        csv_daten = df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            "📥 CSV herunterladen",
            data=csv_daten,
            file_name="fundstuecke.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

st.divider()
st.caption(
    "FundFinder · Streamlit + Hugging Face · "
    "Ohne Datenbank · Sitzungsbasierte Speicherung"
)
