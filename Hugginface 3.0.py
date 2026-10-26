
import io
import json
import base64
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from openai import OpenAI


# ==================================================
# KONFIGURATION
# ==================================================

st.set_page_config(
    page_title="FundFinder – Digitales Fundbüro",
    page_icon="🔎",
    layout="wide",
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

HF_MODELL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


# ==================================================
# SITZUNGSSPEICHER – KEINE DATENBANK
# ==================================================

if "fundstuecke" not in st.session_state:
    st.session_state.fundstuecke = []

if "ki_ergebnis" not in st.session_state:
    st.session_state.ki_ergebnis = None

if "letzter_upload" not in st.session_state:
    st.session_state.letzter_upload = None


# ==================================================
# DESIGN
# ==================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        background: linear-gradient(120deg, #142b4a, #236b78);
        padding: 1.7rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        color: white;
        margin-bottom: .4rem;
    }
    .hero p {
        color: #e4f3f5;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# HUGGING FACE KI
# ==================================================

def hole_hf_token():
    """Liest den Token ausschließlich aus der aktuellen Sitzung."""
    return st.session_state.get("hf_token", "").strip()


def bereite_bild_vor(upload):
    """Korrigiert die Bildausrichtung und verkleinert das Foto."""
    bild = Image.open(upload)
    bild = ImageOps.exif_transpose(bild).convert("RGB")
    bild.thumbnail((1280, 1280))

    puffer = io.BytesIO()
    bild.save(puffer, format="JPEG", quality=85, optimize=True)

    return puffer.getvalue()


def analysiere_foto(bild_bytes):
    """Analysiert ein Foto über die Hugging Face Inference API."""

    token = hole_hf_token()

    if not token:
        raise ValueError(
            "Bitte gib deinen Hugging-Face-Token in der Seitenleiste ein."
        )

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )

    bild_base64 = base64.b64encode(bild_bytes).decode("utf-8")
    bild_url = f"data:image/jpeg;base64,{bild_base64}"

    prompt = f"""
    Du bist eine KI für ein digitales Fundbüro.

    Analysiere den Gegenstand auf dem Foto.
    Beschreibe nur Merkmale, die tatsächlich sichtbar sind.
    Erfinde keine Marke, keinen Fundort und keine persönlichen Angaben.

    Wähle genau eine Kategorie aus:
    {", ".join(KATEGORIEN)}

    Antworte ausschließlich mit einem gültigen JSON-Objekt:

    {{
      "gegenstand": "kurzer Name",
      "kategorie": "eine Kategorie aus der Liste",
      "beschreibung": "sichtbare Merkmale",
      "sicherheit": 0.0
    }}

    Sicherheit ist eine Schätzung zwischen 0 und 1.
    Bei Unsicherheit verwende "Sonstiges".
    """

    antwort = client.chat.completions.create(
        model=HF_MODELL,
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

    text = antwort.choices[0].message.content

    if not text:
        raise ValueError("Die KI hat keine Antwort zurückgegeben.")

    text = text.strip()

    # Eventuelle Markdown-Codeblöcke entfernen
    if text.startswith("```"):
        zeilen = text.splitlines()
        zeilen = zeilen[1:]
        if zeilen and zeilen[-1].strip().startswith("```"):
            zeilen = zeilen[:-1]
        text = "\n".join(zeilen).strip()

    daten = json.loads(text)

    kategorie = daten.get("kategorie", "Sonstiges")
    if kategorie not in KATEGORIEN:
        kategorie = "Sonstiges"

    try:
        sicherheit = float(daten.get("sicherheit", 0))
    except (TypeError, ValueError):
        sicherheit = 0.0

    return {
        "gegenstand": str(
            daten.get("gegenstand", "Unbekannter Gegenstand")
        ),
        "kategorie": kategorie,
        "beschreibung": str(
            daten.get("beschreibung", "Keine Beschreibung")
        ),
        "sicherheit": max(0.0, min(1.0, sicherheit)),
    }


# ==================================================
# KOPFBEREICH
# ==================================================

st.markdown(
    """
    <div class="hero">
        <h1>🔎 FundFinder</h1>
        <p>Dein digitales Fundbüro mit KI-Bilderkennung</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# NAVIGATION UND TOKEN-EINGABE
# ==================================================

with st.sidebar:
    st.title("📍 FundFinder")
    seite = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Fundstück erfassen",
            "Fundstücke durchsuchen",
            "Daten exportieren",
        ],
    )

    st.divider()
    st.subheader("🤖 Hugging Face KI")

    st.text_input(
        "Hugging-Face-Token",
        type="password",
        key="hf_token",
        help="Wird nicht in einer Datei oder Datenbank gespeichert.",
    )

    st.caption(
        "Der Token wird für API-Aufrufe verwendet. "
        "Nach Verlust der Sitzung musst du ihn erneut eingeben."
    )

    st.divider()
    st.caption("Keine Datenbank")
    st.caption("Keine Secrets-Datei")
    st.caption("Sitzungsbasierte Speicherung")


# ==================================================
# DASHBOARD
# ==================================================

if seite == "Dashboard":
    fundstuecke = st.session_state.fundstuecke

    st.subheader("Dashboard")
    st.write("Übersicht der aktuell gespeicherten Fundstücke.")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Gesamt", len(fundstuecke))
    c2.metric(
        "Abgegeben",
        sum(f["status"] == "Abgegeben" for f in fundstuecke),
    )
    c3.metric(
        "Wird gesucht",
        sum(f["status"] == "Wird gesucht" for f in fundstuecke),
    )
    c4.metric(
        "Zurückgegeben",
        sum(f["status"] == "Zurückgegeben" for f in fundstuecke),
    )

    st.divider()
    st.subheader("Neueste Fundstücke")

    if not fundstuecke:
        st.info("Noch keine Fundstücke vorhanden.")
    else:
        for fund in reversed(fundstuecke[-5:]):
            with st.container(border=True):
                bild_spalte, info_spalte = st.columns([1, 3])

                with bild_spalte:
                    st.image(fund["foto"], use_container_width=True)

                with info_spalte:
                    st.markdown(f"**{fund['titel']}**")
                    st.write(f"Kategorie: {fund['kategorie']}")
                    st.write(f"Status: {fund['status']}")
                    st.write(f"Fundort: {fund['ort'] or 'Nicht angegeben'}")
                    st.write(fund["beschreibung"])
                    st.caption(f"Erfasst: {fund['datum']}")


# ==================================================
# NEUES FUNDSTÜCK ERFASSEN
# ==================================================

elif seite == "Fundstück erfassen":
    st.subheader("＋ Fundstück erfassen")

    st.write(
        "Lade ein Foto hoch und lasse die KI einen Vorschlag erstellen. "
        "Du kannst alle Angaben vor dem Speichern bearbeiten."
    )

    upload = st.file_uploader(
        "Foto auswählen",
        type=["jpg", "jpeg", "png", "webp"],
        key="foto_upload",
    )

    bild_bytes = None

    if upload is not None:
        try:
            bild_bytes = bereite_bild_vor(upload)
            st.image(
                bild_bytes,
                caption="Hochgeladenes Foto",
                width=400,
            )
        except Exception as exc:
            st.error(f"Foto konnte nicht verarbeitet werden: {exc}")

    if bild_bytes is None:
        st.session_state.ki_ergebnis = None
        st.session_state.letzter_upload = None
    elif st.session_state.letzter_upload != bild_bytes:
        st.session_state.ki_ergebnis = None
        st.session_state.letzter_upload = bild_bytes

    if bild_bytes is not None:
        if st.button(
            "🤖 Foto mit Hugging Face analysieren",
            type="primary",
            use_container_width=True,
        ):
            if not hole_hf_token():
                st.error(
                    "Gib zuerst deinen Hugging-Face-Token "
                    "in der Seitenleiste ein."
                )
            else:
                with st.spinner("Die KI analysiert das Foto ..."):
                    try:
                        ergebnis = analysiere_foto(bild_bytes)
                        st.session_state.ki_ergebnis = ergebnis

                        # Werte für das Formular vorbereiten
                        st.session_state["eingabe_titel"] = (
                            ergebnis["gegenstand"]
                        )
                        st.session_state["eingabe_kategorie"] = (
                            ergebnis["kategorie"]
                        )
                        st.session_state["eingabe_beschreibung"] = (
                            ergebnis["beschreibung"]
                        )

                        st.success("KI-Analyse abgeschlossen.")

                    except Exception as exc:
                        st.error(
                            "Die Analyse ist fehlgeschlagen. "
                            "Prüfe Token, Modellzugriff und API-Verfügbarkeit."
                        )
                        st.code(str(exc))

    ergebnis = st.session_state.ki_ergebnis

    if ergebnis:
        st.info(
            f"KI-Vorschlag: {ergebnis['gegenstand']} · "
            f"Sicherheit: {ergebnis['sicherheit']:.0%}"
        )

    st.divider()
    st.markdown("### Angaben zum Fundstück")

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
            placeholder="Farbe, Marke und weitere sichtbare Merkmale",
        )

        ort = st.text_input(
            "Fundort",
            placeholder="z. B. Bibliothek, Raum 204",
        )

        funddatum = st.date_input(
            "Funddatum",
            value=datetime.now().date(),
        )

        status = st.selectbox(
            "Status",
            STATUS_OPTIONEN,
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
            st.session_state.fundstuecke.append(
                {
                    "id": uuid.uuid4().hex,
                    "titel": titel.strip(),
                    "kategorie": kategorie,
                    "beschreibung": beschreibung.strip(),
                    "ort": ort.strip(),
                    "funddatum": funddatum.isoformat(),
                    "status": status,
                    "datum": datetime.now().strftime("%d.%m.%Y %H:%M"),
                    "foto": bild_bytes,
                    "ki_sicherheit": (
                        ergebnis["sicherheit"] if ergebnis else None
                    ),
                }
            )

            st.session_state.ki_ergebnis = None
            st.session_state.letzter_upload = None

            st.success("Fundstück wurde gespeichert.")
            st.warning(
                "Das Fundstück ist nur in der aktuellen Sitzung verfügbar. "
                "Es wird nicht dauerhaft gespeichert."
            )


# ==================================================
# SUCHEN, FILTERN UND VERWALTEN
# ==================================================

elif seite == "Fundstücke durchsuchen":
    st.subheader("🔍 Fundstücke durchsuchen")

    fundstuecke = st.session_state.fundstuecke

    if not fundstuecke:
        st.info("Es sind noch keine Fundstücke gespeichert.")
    else:
        suchtext = st.text_input(
            "Suchbegriff",
            placeholder="Gegenstand, Beschreibung oder Fundort",
        )

        col1, col2 = st.columns(2)

        with col1:
            filter_kategorie = st.selectbox(
                "Kategorie",
                ["Alle Kategorien"] + KATEGORIEN,
            )

        with col2:
            filter_status = st.selectbox(
                "Status",
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
                    st.write(f"Kategorie: {fund['kategorie']}")
                    st.write(f"Beschreibung: {fund['beschreibung']}")
                    st.write(f"Fundort: {fund['ort'] or 'Nicht angegeben'}")
                    st.write(f"Funddatum: {fund['funddatum']}")

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

                    if st.button(
                        "🗑️ Fundstück löschen",
                        key=f"loeschen_{fund['id']}",
                    ):
                        st.session_state.fundstuecke = [
                            item
                            for item in st.session_state.fundstuecke
                            if item["id"] != fund["id"]
                        ]
                        st.session_state.pop(status_key, None)
                        st.rerun()


# ==================================================
# CSV-EXPORT
# ==================================================

elif seite == "Daten exportieren":
    st.subheader("⬇️ Daten exportieren")

    fundstuecke = st.session_state.fundstuecke

    if not fundstuecke:
        st.info("Keine Fundstücke zum Exportieren vorhanden.")
    else:
        export_daten = [
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
            }
            for fund in fundstuecke
        ]

        df = pd.DataFrame(export_daten)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "📥 CSV herunterladen",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name="fundstuecke.csv",
            mime="text/csv",
            type="primary",
        )


# ==================================================
# FOOTER
# ==================================================

st.divider()
st.caption(
    "FundFinder · Streamlit + Hugging Face · "
    "Keine Datenbank · Keine Secrets-Datei"
)
