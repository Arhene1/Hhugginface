import os
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
DB_PATH = Path(os.getenv("FUNDBUERO_DB", str(APP_DIR / "fundbuero.db")))
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")

st.set_page_config(page_title="Fundbüro KI", page_icon="🔎", layout="wide")

CATEGORIES = [
    "Schlüssel", "Elektronik", "Tasche/Rucksack", "Kleidung", "Dokumente",
    "Schmuck", "Brille", "Buch/Schreibwaren", "Sport", "Sonstiges"
]
STATUSES = ["Gemeldet", "Zusammengeführt", "Zurückgegeben", "Archiviert"]


def connect():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                location TEXT NOT NULL,
                event_date TEXT NOT NULL,
                contact TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Gemeldet',
                created_at TEXT NOT NULL
            )
        """)
        con.commit()


def get_items(kind=None, include_archived=False):
    query = "SELECT * FROM items WHERE 1=1"
    args = []
    if kind:
        query += " AND kind = ?"
        args.append(kind)
    if not include_archived:
        query += " AND status NOT IN ('Archiviert', 'Zurückgegeben')"
    query += " ORDER BY created_at DESC"
    with connect() as con:
        return [dict(row) for row in con.execute(query, args).fetchall()]


def save_item(data):
    with connect() as con:
        con.execute("""
            INSERT INTO items
            (kind, title, category, description, location, event_date, contact, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'Gemeldet', ?)
        """, (
            data["kind"], data["title"].strip(), data["category"], data["description"].strip(),
            data["location"].strip(), data["event_date"], data["contact"].strip(),
            datetime.now().isoformat(timespec="seconds")
        ))
        con.commit()


def update_status(item_id, status):
    with connect() as con:
        con.execute("UPDATE items SET status = ? WHERE id = ?", (status, int(item_id)))
        con.commit()


def hf_client():
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        try:
            token = str(st.secrets.get("HF_TOKEN", "")).strip()
        except Exception:
            token = ""
    if not token:
        return None
    try:
        from huggingface_hub import InferenceClient
        return InferenceClient(model=HF_MODEL, token=token)
    except Exception:
        return None


def ai_match(lost, found):
    """Ask Hugging Face to assess candidate matches; fall back to transparent keyword scoring."""
    client = hf_client()
    if client:
        prompt = f"""
Du unterstützt ein deutsches Fundbüro. Vergleiche eine Verlustmeldung mit einem Fund.
Bewerte nur anhand der angegebenen Informationen. Erfinde keine Fakten.
Antworte auf Deutsch und kurz mit:
- Wahrscheinlichkeit: hoch/mittel/niedrig
- Gemeinsamkeiten: ...
- Unterschiede/offene Fragen: ...
- Empfehlung: ...
Verlustmeldung:
Titel: {lost['title']}
Kategorie: {lost['category']}
Beschreibung: {lost['description']}
Ort: {lost['location']}
Datum: {lost['event_date']}

Fundmeldung:
Titel: {found['title']}
Kategorie: {found['category']}
Beschreibung: {found['description']}
Ort: {found['location']}
Datum: {found['event_date']}
"""
        try:
            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": "Du bist ein vorsichtiger Assistent für Fundbüro-Abgleiche."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=220,
                temperature=0.2,
            )
            return response.choices[0].message.content, "Hugging Face"
        except Exception as exc:
            ai_error = f"Hugging Face gerade nicht erreichbar ({type(exc).__name__})."
    else:
        ai_error = "Kein HF_TOKEN konfiguriert – es wird eine einfache lokale Ähnlichkeitssuche verwendet."

    import re
    stop = {"der", "die", "das", "und", "oder", "ein", "eine", "mit", "von", "im", "in", "am", "an",
            "ist", "war", "den", "dem", "des", "ich", "habe", "mein", "meine", "gefunden", "verloren"}
    def words(s):
        return {w for w in re.findall(r"[a-zäöüß0-9]+", s.lower()) if len(w) > 2 and w not in stop}
    a = words(" ".join(str(lost.get(k, "")) for k in ("title", "category", "description", "location")))
    b = words(" ".join(str(found.get(k, "")) for k in ("title", "category", "description", "location")))
    shared = sorted(a & b)
    union = a | b
    score = len(shared) / max(len(union), 1)
    level = "hoch" if score >= .35 else "mittel" if score >= .15 else "niedrig"
    result = (
        f"**Wahrscheinlichkeit:** {level} (lokaler Stichwortvergleich: {score:.0%})\n\n"
        f"**Gemeinsame Begriffe:** {', '.join(shared) if shared else 'Keine eindeutigen gemeinsamen Begriffe'}\n\n"
        f"**Hinweis:** {ai_error}\n\n"
        "Bitte den Fund anhand zusätzlicher Merkmale prüfen. Der Score ist keine verlässliche Identitätsprüfung."
    )
    return result, "Lokaler Fallback"


init_db()

st.title("🔎 Fundbüro mit KI-Unterstützung")
st.caption("Verlustmeldungen und Fundstücke verwalten und mögliche Treffer vorschlagen.")

with st.sidebar:
    st.header("Übersicht")
    all_active = get_items(include_archived=False)
    st.metric("Aktive Meldungen", len(all_active))
    st.metric("Verlustmeldungen", sum(x["kind"] == "Verlust" for x in all_active))
    st.metric("Fundmeldungen", sum(x["kind"] == "Fund" for x in all_active))
    st.divider()
    st.caption("KI-Modell")
    st.code(HF_MODEL, language=None)
    st.caption("Für die Hugging Face API muss HF_TOKEN gesetzt sein. Ohne Token funktioniert die lokale Stichwortsuche.")

tab_new, tab_search, tab_match, tab_admin = st.tabs([
    "➕ Meldung erfassen", "📋 Meldungen durchsuchen", "🤖 KI-Abgleich", "🛠️ Verwaltung"
])

with tab_new:
    st.subheader("Neue Verlust- oder Fundmeldung")
    with st.form("new_item_form", clear_on_submit=True):
        kind_label = st.radio("Art der Meldung", ["Ich habe etwas verloren", "Ich habe etwas gefunden"], horizontal=True)
        kind = "Verlust" if kind_label.startswith("Ich habe etwas verloren") else "Fund"
        c1, c2 = st.columns(2)
        with c1:
            title = st.text_input("Bezeichnung *", placeholder="z. B. schwarzer Rucksack")
            category = st.selectbox("Kategorie", CATEGORIES)
            location = st.text_input("Ort *", placeholder="z. B. Bibliothek, Raum 204")
            event_date = st.date_input("Datum des Verlusts/Funds", value=datetime.now().date())
        with c2:
            description = st.text_area(
                "Beschreibung *",
                placeholder="Farbe, Marke, besondere Merkmale – keine sensiblen Daten eintragen.",
                height=140
            )
            contact = st.text_input("Kontakt für Rückfragen *", placeholder="E-Mail oder Telefonnummer")
        privacy_ok = st.checkbox("Ich habe verstanden, dass die Angaben lokal in der Datenbank gespeichert werden.")
        submitted = st.form_submit_button("Meldung speichern", type="primary")
        if submitted:
            if not all([title.strip(), location.strip(), description.strip(), contact.strip()]):
                st.error("Bitte alle Pflichtfelder (*) ausfüllen.")
            elif not privacy_ok:
                st.warning("Bitte die Speicherbestätigung aktivieren.")
            else:
                save_item({
                    "kind": kind, "title": title, "category": category,
                    "description": description, "location": location,
                    "event_date": event_date.isoformat(), "contact": contact
                })
                st.success("Meldung wurde gespeichert.")
                st.rerun()

with tab_search:
    st.subheader("Meldungen")
    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        kind_filter = st.selectbox("Art", ["Alle", "Verlust", "Fund"])
    with f2:
        category_filter = st.selectbox("Kategorie", ["Alle"] + CATEGORIES)
    with f3:
        search_text = st.text_input("Suchbegriff", placeholder="Bezeichnung, Beschreibung oder Ort")
    items = get_items(kind=None if kind_filter == "Alle" else kind_filter)
    if category_filter != "Alle":
        items = [x for x in items if x["category"] == category_filter]
    if search_text.strip():
        needle = search_text.lower()
        items = [x for x in items if needle in " ".join(str(x[k]) for k in ("title", "description", "location", "category")).lower()]
    if not items:
        st.info("Keine passenden aktiven Meldungen gefunden.")
    else:
        for item in items:
            label = f"#{item['id']} · {item['kind']} · {item['title']}"
            with st.expander(label):
                a, b = st.columns([3, 1])
                with a:
                    st.write(f"**Kategorie:** {item['category']}")
                    st.write(f"**Beschreibung:** {item['description']}")
                    st.write(f"**Ort:** {item['location']}")
                    st.write(f"**Datum:** {item['event_date']}")
                    st.write(f"**Status:** {item['status']}")
                    st.caption("Kontaktangaben werden in dieser Demo nur in der Detailansicht angezeigt. Schütze den Zugriff auf die App.")
                    st.write(f"**Kontakt:** {item['contact']}")
                with b:
                    st.write(f"**Erstellt:** {item['created_at'][:10]}")
                    new_status = st.selectbox(
                        "Status ändern", STATUSES, index=STATUSES.index(item["status"]),
                        key=f"status_{item['id']}"
                    )
                    if st.button("Status speichern", key=f"save_status_{item['id']}"):
                        update_status(item["id"], new_status)
                        st.rerun()

with tab_match:
    st.subheader("Mögliche Treffer finden")
    losses = get_items(kind="Verlust")
    finds = get_items(kind="Fund")
    if not losses or not finds:
        st.info("Für den Abgleich brauchst du mindestens eine aktive Verlust- und eine Fundmeldung.")
    else:
        loss_options = {f"#{x['id']} · {x['title']} ({x['location']})": x for x in losses}
        selected_loss_label = st.selectbox("Verlustmeldung auswählen", list(loss_options))
        lost = loss_options[selected_loss_label]
        st.markdown("**Passende Fundmeldungen vergleichen**")
        candidate_rows = []
        for found in finds:
            same_category = found["category"] == lost["category"]
            same_location = found["location"].strip().lower() == lost["location"].strip().lower()
            candidate_rows.append((found, same_category, same_location))
        candidate_rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
        candidate_labels = {
            f"#{f['id']} · {f['title']} · {f['category']} · {f['location']}": f
            for f, _, _ in candidate_rows
        }
        selected_found_label = st.selectbox("Fundmeldung auswählen", list(candidate_labels))
        found = candidate_labels[selected_found_label]
        with st.expander("Details der ausgewählten Meldungen", expanded=False):
            st.write("**Verlust:**", lost["description"])
            st.write("**Fund:**", found["description"])
        if st.button("🤖 Mit KI vergleichen", type="primary"):
            with st.spinner("Vergleich wird erstellt …"):
                result, engine = ai_match(lost, found)
            st.caption(f"Vergleichsmethode: {engine}")
            st.markdown(result)
            st.warning("Die KI liefert nur einen Hinweis. Sie kann falsch liegen. Vor einer Rückgabe muss die berechtigte Person den Gegenstand anhand nicht öffentlich genannter Merkmale bestätigen.")

with tab_admin:
    st.subheader("Datenverwaltung")
    st.warning("Diese Demo speichert Daten in einer lokalen SQLite-Datei. Es gibt keine Benutzeranmeldung oder Rollenverwaltung. Nicht ungeschützt öffentlich bereitstellen.")
    include_archived = st.checkbox("Zurückgegebene und archivierte Meldungen anzeigen", value=True)
    admin_items = get_items(include_archived=include_archived)
    if admin_items:
        df = pd.DataFrame(admin_items)
        st.dataframe(
            df[["id", "kind", "title", "category", "location", "event_date", "status", "created_at"]],
            use_container_width=True,
            hide_index=True
        )
        csv_data = df.drop(columns=["contact"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
        st.download_button("Meldungen als CSV exportieren", csv_data, "fundbuero_meldungen.csv", "text/csv")
    else:
        st.info("Noch keine Meldungen vorhanden.")
    st.caption("Tipp: Sichere die Datei fundbuero.db regelmäßig und beachte Datenschutz- und Löschfristen.")
