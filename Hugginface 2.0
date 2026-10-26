
import os
import sqlite3
import uuid
from pathlib import Path
from datetime import date, datetime

import pandas as pd
import streamlit as st
from PIL import Image
from huggingface_hub import InferenceClient

# --------------------------------------------------
# KONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PHOTO_DIR = DATA_DIR / "fotos"
DB_PATH = DATA_DIR / "fundbuero.db"

DATA_DIR.mkdir(exist_ok=True)
PHOTO_DIR.mkdir(exist_ok=True)

CATEGORIES = [
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

STATUSES = [
    "Neu",
    "Aufbewahrt",
    "An Eigentümer übergeben",
    "Entsorgt",
]

st.set_page_config(
    page_title="Fundbüro KI",
    page_icon="🔎",
    layout="wide",
)

# --------------------------------------------------
# DATENBANK
# --------------------------------------------------

def get_connection():
    return sqlite3.connect(DB_PATH)


def init_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS funds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT DEFAULT '',
                location TEXT DEFAULT '',
                found_date TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Neu',
                photo_path TEXT,
                ai_label TEXT,
                ai_score REAL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


def get_funds(search="", category="Alle", status="Alle"):
    query = """
        SELECT * FROM funds
        WHERE 1=1
    """
    params = []

    if search.strip():
        query += """
            AND (
                title LIKE ?
                OR description LIKE ?
                OR location LIKE ?
                OR ai_label LIKE ?
            )
        """
        term = f"%{search.strip()}%"
        params.extend([term, term, term, term])

    if category != "Alle":
        query += " AND category = ?"
        params.append(category)

    if status != "Alle":
        query += " AND status = ?"
        params.append(status)

    query += " ORDER BY found_date DESC, id DESC"

    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(query, params).fetchall()


# --------------------------------------------------
# HUGGING FACE KI
# --------------------------------------------------

def classify_image(image_bytes):
    token = os.getenv("HF_TOKEN", "").strip()

    if not token:
        raise ValueError(
            "Der Hugging-Face-Token fehlt. "
            "Bitte die Umgebungsvariable HF_TOKEN setzen."
        )

    model = os.getenv(
        "HF_IMAGE_MODEL",
        "google/vit-base-patch16-224",
    )

    client = InferenceClient(
        provider="hf-inference",
        api_key=token,
    )

    predictions = client.image_classification(
        image_bytes,
        model=model,
    )

    if not predictions:
        raise ValueError("Die KI hat keine Ergebnisse geliefert.")

    best = predictions[0]

    return best.label, float(best.score)


def suggest_category(label):
    """Ordnet typische englische Bildlabels einer Fundkategorie zu."""

    label = label.lower().replace("_", " ").replace("-", " ")

    # Spezifische Regeln zuerst
    rules = [
        (["wallet", "purse", "coin purse"], "Geldbörse"),
        (["key", "keychain"], "Schlüssel"),
        (
            ["backpack", "handbag", "suitcase", "bag"],
            "Tasche / Rucksack",
        ),
        (
            [
                "cellular telephone", "mobile phone", "telephone",
                "laptop", "computer", "remote control",
                "headphone", "mouse", "keyboard", "camera",
            ],
            "Elektronik",
        ),
        (
            ["sunglass", "eyeglass", "spectacle", "glasses"],
            "Brille",
        ),
        (["umbrella"], "Schirm"),
        (
            ["watch", "ring", "necklace", "earring", "jewelry"],
            "Schmuck / Uhr",
        ),
        (
            [
                "jean", "jersey", "coat", "shirt", "sock",
                "shoe", "sandal", "jacket", "dress", "trouser",
            ],
            "Kleidung",
        ),
        (
            ["bottle", "water jug", "cup", "mug", "flask"],
            "Flasche / Trinkgefäß",
        ),
        (
            ["envelope", "letter", "book jacket"],
            "Dokumente / Ausweise",
        ),
        (
            ["toy", "teddy", "doll", "ball"],
            "Spielzeug",
        ),
    ]

    for keywords, category in rules:
        if any(word in label for word in keywords):
            return category

    return "Sonstiges"


# --------------------------------------------------
# FOTO SPEICHERN
# --------------------------------------------------

def save_photo(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")

    filename = f"{uuid.uuid4().hex}.jpg"
    path = PHOTO_DIR / filename

    image.save(
        path,
        format="JPEG",
        quality=90,
        optimize=True,
    )

    return str(path)


# --------------------------------------------------
# APP STARTEN
# --------------------------------------------------

init_database()

st.title("🔎 Fundbüro mit KI")
st.caption(
    "Fundsachen digital erfassen, kategorisieren und wiederfinden."
)

with get_connection() as conn:
    total = conn.execute(
        "SELECT COUNT(*) FROM funds"
    ).fetchone()[0]

    open_items = conn.execute(
        """
        SELECT COUNT(*) FROM funds
        WHERE status IN ('Neu', 'Aufbewahrt')
        """
    ).fetchone()[0]

    returned = conn.execute(
        """
        SELECT COUNT(*) FROM funds
        WHERE status = 'An Eigentümer übergeben'
        """
    ).fetchone()[0]

m1, m2, m3 = st.columns(3)
m1.metric("Fundsachen gesamt", total)
m2.metric("Noch im Fundbüro", open_items)
m3.metric("Zurückgegeben", returned)

tab_add, tab_search, tab_info = st.tabs([
    "➕ Fund erfassen",
    "📋 Fundregister",
    "⚙️ Einstellungen",
])

# --------------------------------------------------
# NEUE FUNDSACHE
# --------------------------------------------------

with tab_add:
    st.subheader("Neue Fundsache aufnehmen")

    uploaded = st.file_uploader(
        "Foto der Fundsache hochladen",
        type=["jpg", "jpeg", "png", "webp"],
        key="new_photo",
    )

    if uploaded:
        st.image(
            uploaded,
            caption="Hochgeladenes Foto",
            width=300,
        )

    if st.button(
        "🤖 Foto mit Hugging Face analysieren",
        disabled=uploaded is None,
    ):
        try:
            with st.spinner("KI analysiert das Foto ..."):
                label, score = classify_image(uploaded.getvalue())

            st.session_state["ai_label"] = label
            st.session_state["ai_score"] = score
            st.session_state["ai_category"] = suggest_category(label)

            st.success(
                f"Erkannt: {label} "
                f"(Modell-Score: {score:.1%})"
            )

        except Exception as exc:
            st.error(f"KI-Analyse fehlgeschlagen: {exc}")

    ai_label = st.session_state.get("ai_label", "")
    ai_score = st.session_state.get("ai_score")
    ai_category = st.session_state.get(
        "ai_category", "Sonstiges"
    )

    if ai_label:
        st.info(
            f"KI-Vorschlag: **{ai_label}** · "
            f"Kategorie: **{ai_category}**"
        )

    with st.form("new_fund_form"):
        title = st.text_input(
            "Bezeichnung *",
            placeholder="z. B. Schwarzer Schlüsselbund",
        )

        category = st.selectbox(
            "Kategorie",
            CATEGORIES,
            index=(
                CATEGORIES.index(ai_category)
                if ai_category in CATEGORIES
                else CATEGORIES.index("Sonstiges")
            ),
        )

        description = st.text_area(
            "Beschreibung",
            placeholder="Farbe, Marke, besondere Merkmale ...",
        )

        location = st.text_input(
            "Fundort",
            placeholder="z. B. Haupteingang",
        )

        found_date = st.date_input(
            "Funddatum",
            value=date.today(),
            format="DD.MM.YYYY",
        )

        status = st.selectbox("Status", STATUSES)

        submitted = st.form_submit_button(
            "Fundsache speichern",
            type="primary",
        )

        if submitted:
            if not title.strip():
                st.error("Bitte eine Bezeichnung eingeben.")

            else:
                photo_path = None

                if uploaded:
                    try:
                        photo_path = save_photo(uploaded)
                    except Exception as exc:
                        st.error(f"Foto konnte nicht gespeichert werden: {exc}")
                        st.stop()

                with get_connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO funds (
                            title, category, description, location,
                            found_date, status, photo_path,
                            ai_label, ai_score, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            title.strip(),
                            category,
                            description.strip(),
                            location.strip(),
                            found_date.isoformat(),
                            status,
                            photo_path,
                            ai_label or None,
                            ai_score,
                            datetime.now().isoformat(timespec="seconds"),
                        ),
                    )
                    conn.commit()

                for key in (
                    "ai_label",
                    "ai_score",
                    "ai_category",
                ):
                    st.session_state.pop(key, None)

                st.success("Fundsache erfolgreich gespeichert.")

# --------------------------------------------------
# FUNDREGISTER / SUCHE
# --------------------------------------------------

with tab_search:
    st.subheader("Fundsachen suchen und verwalten")

    c1, c2, c3 = st.columns([2, 1, 1])

    search = c1.text_input(
        "Suchbegriff",
        placeholder="Name, Beschreibung, Fundort ...",
    )

    category_filter = c2.selectbox(
        "Kategorie",
        ["Alle"] + CATEGORIES,
    )

    status_filter = c3.selectbox(
        "Status",
        ["Alle"] + STATUSES,
    )

    funds = get_funds(
        search,
        category_filter,
        status_filter,
    )

    if not funds:
        st.info("Keine passenden Fundsachen gefunden.")

    for fund in funds:
        with st.container(border=True):
            left, right = st.columns([1, 3])

            with left:
                photo = fund["photo_path"]

                if photo and Path(photo).exists():
                    st.image(
                        photo,
                        use_container_width=True,
                    )
                else:
                    st.caption("Kein Foto vorhanden")

            with right:
                st.markdown(
                    f"### #{fund['id']} – {fund['title']}"
                )

                st.write(
                    f"**Kategorie:** {fund['category']}  \n"
                    f"**Status:** {fund['status']}  \n"
                    f"**Funddatum:** {fund['found_date']}  \n"
                    f"**Fundort:** {fund['location'] or 'Nicht angegeben'}"
                )

                if fund["description"]:
                    st.write(fund["description"])

                if fund["ai_label"]:
                    score = fund["ai_score"]
                    score_text = (
                        f" ({score:.1%})"
                        if score is not None
                        else ""
                    )
                    st.caption(
                        f"KI-Erkennung: {fund['ai_label']}{score_text}"
                    )

                with st.expander("Bearbeiten oder löschen"):
                    with st.form(f"edit_{fund['id']}"):
                        new_title = st.text_input(
                            "Bezeichnung",
                            value=fund["title"],
                        )

                        old_category = fund["category"]
                        new_category = st.selectbox(
                            "Kategorie",
                            CATEGORIES,
                            index=(
                                CATEGORIES.index(old_category)
                                if old_category in CATEGORIES
                                else 0
                            ),
                            key=f"cat_{fund['id']}",
                        )

                        new_description = st.text_area(
                            "Beschreibung",
                            value=fund["description"] or "",
                        )

                        new_location = st.text_input(
                            "Fundort",
                            value=fund["location"] or "",
                        )

                        try:
                            old_date = date.fromisoformat(
                                fund["found_date"]
                            )
                        except ValueError:
                            old_date = date.today()

                        new_date = st.date_input(
                            "Funddatum",
                            value=old_date,
                            key=f"date_{fund['id']}",
                            format="DD.MM.YYYY",
                        )

                        old_status = fund["status"]
                        new_status = st.selectbox(
                            "Status",
                            STATUSES,
                            index=(
                                STATUSES.index(old_status)
                                if old_status in STATUSES
                                else 0
                            ),
                            key=f"status_{fund['id']}",
                        )

                        save_changes = st.form_submit_button(
                            "Änderungen speichern"
                        )

                        if save_changes:
                            if not new_title.strip():
                                st.error("Die Bezeichnung darf nicht leer sein.")
                            else:
                                with get_connection() as conn:
                                    conn.execute(
                                        """
                                        UPDATE funds
                                        SET title = ?,
                                            category = ?,
                                            description = ?,
                                            location = ?,
                                            found_date = ?,
                                            status = ?
                                        WHERE id = ?
                                        """,
                                        (
                                            new_title.strip(),
                                            new_category,
                                            new_description.strip(),
                                            new_location.strip(),
                                            new_date.isoformat(),
                                            new_status,
                                            fund["id"],
                                        ),
                                    )
                                    conn.commit()

                                st.success("Änderungen gespeichert.")
                                st.rerun()

                    if st.button(
                        "🗑️ Fundsache löschen",
                        key=f"delete_{fund['id']}",
                    ):
                        with get_connection() as conn:
                            conn.execute(
                                "DELETE FROM funds WHERE id = ?",
                                (fund["id"],),
                            )
                            conn.commit()

                        if fund["photo_path"]:
                            try:
                                Path(fund["photo_path"]).unlink(
                                    missing_ok=True
                                )
                            except OSError:
                                pass

                        st.rerun()

    if funds:
        export_data = pd.DataFrame([
            dict(fund) for fund in funds
        ]).drop(
            columns=["photo_path"],
            errors="ignore",
        )

        st.download_button(
            "⬇️ Suchergebnis als CSV exportieren",
            data=export_data.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="fundregister.csv",
            mime="text/csv",
        )

# --------------------------------------------------
# HINWEISE
# --------------------------------------------------

with tab_info:
    st.subheader("Konfiguration und Datenschutz")

    st.markdown("""
    **KI-Modell**

    Standardmäßig wird `google/vit-base-patch16-224` verwendet.
    Das Modell ist ein allgemeines Bildklassifikationsmodell.
    Es kann Gegenstände falsch erkennen. Prüfe deshalb immer
    den KI-Vorschlag, bevor du einen Datensatz speicherst.

    **Hugging Face Token**

    Der Zugangsschlüssel wird über die Umgebungsvariable
    `HF_TOKEN` geladen und nicht im Quellcode gespeichert.
    Dein Token benötigt die erforderlichen Berechtigungen
    für Inference Providers.

    **Datenschutz**

    Für die KI-Analyse wird das Foto an Hugging Face übermittelt.
    Verwende nur Fotos, deren Übertragung zulässig ist.
    Vermeide unnötige personenbezogene Daten, Gesichter
    und lesbare Ausweise.

    **Speicherung**

    Die Datenbank und die Fotos werden lokal im Verzeichnis
    `data/` gespeichert. Bei einer Neuinstallation oder einem
    Serverumzug muss dieses Verzeichnis gesichert werden.

    **Vor produktivem Einsatz**

    Diese Version hat noch keine Benutzerverwaltung,
    Rollen- und Rechteverwaltung, automatischen Backups
    oder Schutz vor gleichzeitigen Bearbeitungskonflikten.
    Für einen echten öffentlichen oder betrieblichen Einsatz
    sollte sie entsprechend abgesichert werden.
    """)

    st.markdown(
        "[Hugging Face Inference Providers – Dokumentation]"
        "(https://huggingface.co/docs/inference-providers)"
    )
