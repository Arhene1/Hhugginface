import os
import json
import base64
import tempfile
from pathlib import Path
from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "fundfinder_data.json"
UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="FundFinder", page_icon="🔎", layout="wide")

CATEGORIES = [
    "Schlüssel", "Geldbörse", "Handy / Elektronik", "Tasche / Rucksack",
    "Kleidung", "Dokumente", "Schmuck", "Fahrrad", "Sonstiges"
]
STATUSES = ["Noch nicht abgeholt", "Zurückgegeben"]


def load_items():
    """Load shared JSON data. File storage is convenient but ephemeral on Streamlit Cloud."""
    if not DATA_FILE.exists():
        return []
    try:
        with DATA_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        st.warning("Die lokale Fundstück-Datei konnte nicht gelesen werden. Es wird eine leere Liste angezeigt.")
        return []


def save_items(items):
    """Write atomically to reduce the chance of a partially written JSON file."""
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=str(APP_DIR), delete=False, suffix=".tmp"
        ) as tmp:
            json.dump(items, tmp, ensure_ascii=False, indent=2)
            temp_name = tmp.name
        os.replace(temp_name, DATA_FILE)
        return True
    except OSError as exc:
        st.error(f"Speichern fehlgeschlagen: {exc}")
        return False


def get_hf_token():
    """Token is optional. It is never requested from visitors or shown in the UI."""
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        return str(st.secrets.get("HF_TOKEN", "")).strip()
    except Exception:
        return ""


def analyze_image(image_bytes, filename):
    """
    Optional image recognition through Hugging Face's OpenAI-compatible router.
    If credentials/model/provider are unavailable, return a useful manual fallback.
    """
    token = get_hf_token()
    if not token:
        return None, "KI-Erkennung ist nicht eingerichtet. Bitte Kategorie und Beschreibung manuell eingeben."

    try:
        from openai import OpenAI
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=token,
            timeout=30.0,
            max_retries=1,
        )
        response = client.chat.completions.create(
            model="Qwen/Qwen3-VL-30B-A3B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analysiere das Foto eines verlorenen oder gefundenen Gegenstands. "
                                "Antworte auf Deutsch in höchstens 3 kurzen Sätzen. "
                                "Nenne zuerst eine passende Kategorie aus dieser Liste: "
                                + ", ".join(CATEGORIES)
                                + ". Danach beschreibe sichtbare Merkmale. Keine Vermutungen über Besitzer."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                        },
                    ],
                }
            ],
            max_tokens=160,
        )
        result = (response.choices[0].message.content or "").strip()
        if not result:
            return None, "Die KI hat keine Beschreibung zurückgegeben. Bitte manuell ausfüllen."
        return result, None
    except Exception as exc:
        message = str(exc)
        if "401" in message or "Unauthorized" in message:
            return None, "Hugging Face hat den Zugang abgelehnt (401). Prüfe den HF_TOKEN in der Server-Konfiguration."
        if "403" in message or "Forbidden" in message:
            return None, "Der Token hat keine Berechtigung für dieses Modell oder diesen Provider (403)."
        if "404" in message or "not found" in message.lower():
            return None, "Das Modell oder der Provider ist aktuell nicht verfügbar. Bitte manuell beschreiben."
        return None, f"KI-Erkennung nicht verfügbar. Du kannst das Fundstück trotzdem speichern. Details: {message[:220]}"


def save_uploaded_image(uploaded_file, item_id):
    if not uploaded_file:
        return ""
    try:
        image = Image.open(uploaded_file).convert("RGB")
        path = UPLOAD_DIR / f"{item_id}.jpg"
        image.save(path, format="JPEG", quality=88)
        return str(path.relative_to(APP_DIR))
    except Exception:
        return ""


def safe_image_path(relative_path):
    if not relative_path:
        return None
    candidate = (APP_DIR / relative_path).resolve()
    try:
        candidate.relative_to(APP_DIR.resolve())
    except ValueError:
        return None
    return candidate if candidate.exists() else None


st.title("🔎 FundFinder")
st.caption("Fundstücke erfassen, durchsuchen und den Rückgabe-Status verwalten.")

with st.sidebar:
    st.header("Über die App")
    st.write("Die KI-Bilderkennung ist optional. Ohne eingerichteten HF_TOKEN kannst du Fundstücke manuell erfassen.")
    st.info(
        "Hinweis zum Speicher: Diese Version verwendet eine JSON-Datei statt einer Datenbank. "
        "Auf Streamlit Community Cloud kann diese Datei bei Neustarts oder erneuten Deployments verloren gehen. "
        "Für dauerhaft geteilte Daten ist ein persistenter gemeinsamer Speicher nötig."
    )

items = load_items()
tab_overview, tab_add, tab_search, tab_export = st.tabs(
    ["📊 Übersicht", "➕ Fundstück erfassen", "🔍 Suchen & verwalten", "📤 Export"]
)

with tab_overview:
    col1, col2, col3 = st.columns(3)
    col1.metric("Fundstücke insgesamt", len(items))
    col2.metric("Noch nicht abgeholt", sum(x.get("status") == STATUSES[0] for x in items))
    col3.metric("Zurückgegeben", sum(x.get("status") == STATUSES[1] for x in items))
    st.subheader("Zuletzt erfasste Fundstücke")
    if not items:
        st.write("Noch keine Fundstücke erfasst.")
    else:
        recent = sorted(items, key=lambda x: x.get("created_at", ""), reverse=True)[:8]
        for item in recent:
            with st.container(border=True):
                c1, c2 = st.columns([1, 4])
                img_path = safe_image_path(item.get("image_path", ""))
                if img_path:
                    c1.image(str(img_path), use_container_width=True)
                c2.markdown(f"**{item.get('title', 'Ohne Titel')}**")
                c2.write(f"**Kategorie:** {item.get('category', 'Sonstiges')} · **Status:** {item.get('status', STATUSES[0])}")
                c2.write(item.get("description", ""))
                c2.caption(f"Ort: {item.get('location') or 'Nicht angegeben'} · Erfasst: {item.get('created_at', '')}")

with tab_add:
    st.subheader("Neues Fundstück")
    with st.form("add_item_form", clear_on_submit=True):
        title = st.text_input("Titel *", placeholder="z. B. Schwarze Geldbörse")
        location = st.text_input("Fundort", placeholder="z. B. Bibliothek, Raum 2.14")
        found_date = st.date_input("Funddatum", value=datetime.now().date())
        uploaded = st.file_uploader("Foto (optional)", type=["png", "jpg", "jpeg", "webp"])
        category = st.selectbox("Kategorie", CATEGORIES)
        description = st.text_area("Beschreibung", placeholder="Farbe, Marke, besondere Merkmale …")
        run_ai = st.checkbox("Foto mit Hugging Face analysieren (wenn eingerichtet)", value=True)
        submitted = st.form_submit_button("Fundstück speichern", type="primary")

    if submitted:
        if not title.strip():
            st.error("Bitte gib einen Titel ein.")
        else:
            ai_note = ""
            if uploaded and run_ai:
                image_bytes = uploaded.getvalue()
                with st.spinner("Foto wird analysiert …"):
                    ai_result, ai_error = analyze_image(image_bytes, uploaded.name)
                if ai_result:
                    ai_note = ai_result
                    if not description.strip():
                        description = ai_result
                    st.success("KI-Analyse abgeschlossen.")
                elif ai_error:
                    st.warning(ai_error)

            item_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
            image_path = save_uploaded_image(uploaded, item_id) if uploaded else ""
            item = {
                "id": item_id,
                "title": title.strip(),
                "location": location.strip(),
                "found_date": str(found_date),
                "category": category,
                "description": description.strip(),
                "ai_note": ai_note,
                "image_path": image_path,
                "status": STATUSES[0],
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            # Reload immediately before saving so we don't overwrite a newer list unnecessarily.
            latest = load_items()
            latest.append(item)
            if save_items(latest):
                st.success("Fundstück gespeichert.")
                st.rerun()

with tab_search:
    st.subheader("Fundstücke suchen")
    col1, col2, col3 = st.columns(3)
    query = col1.text_input("Suchbegriff", placeholder="Titel, Beschreibung, Fundort …")
    category_filter = col2.selectbox("Kategorie", ["Alle"] + CATEGORIES)
    status_filter = col3.selectbox("Status", ["Alle"] + STATUSES)

    filtered = []
    for item in items:
        haystack = " ".join([
            str(item.get("title", "")), str(item.get("description", "")),
            str(item.get("location", "")), str(item.get("category", ""))
        ]).casefold()
        if query.casefold() not in haystack:
            continue
        if category_filter != "Alle" and item.get("category") != category_filter:
            continue
        if status_filter != "Alle" and item.get("status") != status_filter:
            continue
        filtered.append(item)

    st.caption(f"{len(filtered)} Ergebnis(se)")
    if not filtered:
        st.info("Keine passenden Fundstücke gefunden.")
    for item in sorted(filtered, key=lambda x: x.get("created_at", ""), reverse=True):
        with st.container(border=True):
            left, right = st.columns([1, 3])
            img_path = safe_image_path(item.get("image_path", ""))
            if img_path:
                left.image(str(img_path), use_container_width=True)
            right.markdown(f"### {item.get('title', 'Ohne Titel')}")
            right.write(f"**Kategorie:** {item.get('category', 'Sonstiges')} · **Status:** {item.get('status', STATUSES[0])}")
            right.write(f"**Fundort:** {item.get('location') or 'Nicht angegeben'} · **Funddatum:** {item.get('found_date', '')}")
            right.write(item.get("description", ""))
            with right.expander("Eintrag bearbeiten / löschen"):
                new_status = st.selectbox(
                    "Status ändern",
                    STATUSES,
                    index=STATUSES.index(item.get("status", STATUSES[0])) if item.get("status", STATUSES[0]) in STATUSES else 0,
                    key=f"status_{item['id']}",
                )
                new_description = st.text_area("Beschreibung ändern", value=item.get("description", ""), key=f"desc_{item['id']}")
                b1, b2 = st.columns(2)
                if b1.button("Änderungen speichern", key=f"save_{item['id']}"):
                    latest = load_items()
                    for entry in latest:
                        if entry.get("id") == item["id"]:
                            entry["status"] = new_status
                            entry["description"] = new_description
                    if save_items(latest):
                        st.success("Änderungen gespeichert.")
                        st.rerun()
                if b2.button("Eintrag löschen", key=f"delete_{item['id']}"):
                    latest = [entry for entry in load_items() if entry.get("id") != item["id"]]
                    if save_items(latest):
                        image_file = safe_image_path(item.get("image_path", ""))
                        if image_file:
                            try:
                                image_file.unlink()
                            except OSError:
                                pass
                        st.success("Eintrag gelöscht.")
                        st.rerun()

with tab_export:
    st.subheader("Daten exportieren")
    items = load_items()
    if items:
        export_df = pd.DataFrame(items)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("CSV herunterladen", data=csv_bytes, file_name="fundfinder_fundstuecke.csv", mime="text/csv")
        json_bytes = json.dumps(items, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("JSON-Backup herunterladen", data=json_bytes, file_name="fundfinder_backup.json", mime="application/json")
    else:
        st.info("Es gibt noch keine Daten zum Exportieren.")
