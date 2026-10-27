import base64
import io
import json
import uuid
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="FundFinder", page_icon="🔎", layout="wide")

# This version intentionally uses session state only (no database).
# Streamlit Community Cloud does not guarantee durable/shared local-file storage.
if "items" not in st.session_state:
    st.session_state.items = []
if "hf_token" not in st.session_state:
    st.session_state.hf_token = ""

MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
API_URL = "https://router.huggingface.co/v1/chat/completions"

def image_to_data_url(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image.thumbnail((1200, 1200))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}", image

def classify_image(uploaded_file, token):
    if not token.strip():
        return None, "Kein Hugging-Face-Token angegeben. Du kannst das Fundstück trotzdem manuell speichern."
    try:
        data_url, _ = image_to_data_url(uploaded_file)
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {token.strip()}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Analysiere das Foto eines verlorenen Gegenstands. "
                                    "Antworte auf Deutsch und kurz mit: Gegenstand, Farbe, "
                                    "erkennbare Marke/Details und 3 Suchbegriffe. "
                                    "Erfinde keine Details, die nicht sichtbar sind."
                                ),
                            },
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                "max_tokens": 220,
                "temperature": 0.2,
            },
            timeout=60,
        )
        if response.status_code == 401:
            return None, "Hugging Face meldet 401 (nicht autorisiert). Bitte Token prüfen. Das Speichern ohne KI funktioniert weiterhin."
        if response.status_code == 403:
            return None, "Hugging Face meldet 403 (Zugriff verweigert). Prüfe Token-Berechtigungen und Modellzugriff."
        if response.status_code == 429:
            return None, "Hugging Face meldet 429 (Limit erreicht). Versuche es später erneut."
        if not response.ok:
            detail = response.text[:400].replace("\n", " ")
            return None, f"Hugging-Face-Fehler {response.status_code}: {detail}"
        payload = response.json()
        text = payload["choices"][0]["message"]["content"]
        if isinstance(text, list):
            text = "\n".join(part.get("text", "") for part in text if isinstance(part, dict))
        return str(text).strip(), None
    except requests.Timeout:
        return None, "Zeitüberschreitung bei der KI-Anfrage. Du kannst das Fundstück ohne KI speichern."
    except requests.RequestException as exc:
        return None, f"Netzwerkfehler bei der KI-Anfrage: {exc}"
    except (ValueError, KeyError, IndexError) as exc:
        return None, f"Unerwartete Antwort der KI: {exc}"
    except Exception as exc:
        return None, f"Foto konnte nicht verarbeitet werden: {exc}"

def reset_form():
    for key in ("title", "description", "location", "contact"):
        st.session_state[key] = ""

st.title("🔎 FundFinder")
st.caption("Fundstücke erfassen, durchsuchen und verwalten – die KI-Bilderkennung ist optional.")

with st.sidebar:
    st.header("Einstellungen")
    st.info(
        "Ohne Token läuft die Fundverwaltung weiterhin. Für die optionale KI-Erkennung "
        "kannst du hier deinen Hugging-Face-Token eingeben. Er wird nur in deiner "
        "aktuellen Streamlit-Sitzung verwendet und nicht in dieser App gespeichert."
    )
    st.text_input("Hugging-Face-Token (optional)", type="password", key="hf_token")
    st.caption("Teile deinen Token nicht und committe ihn niemals in GitHub.")
    st.divider()
    st.metric("Fundstücke in dieser Sitzung", len(st.session_state.items))
    st.warning(
        "Diese Version verwendet keinen gemeinsamen Speicher. Andere Nutzer sehen "
        "nicht automatisch deine Einträge; Einträge können beim Neustart verloren gehen."
    )

tab_dashboard, tab_add, tab_search, tab_export = st.tabs(
    ["📊 Übersicht", "➕ Fundstück melden", "🔍 Suchen & verwalten", "📥 Export"]
)

with tab_dashboard:
    items = st.session_state.items
    total = len(items)
    open_count = sum(1 for item in items if item["status"] == "Offen")
    returned_count = sum(1 for item in items if item["status"] == "Zurückgegeben")
    c1, c2, c3 = st.columns(3)
    c1.metric("Alle Fundstücke", total)
    c2.metric("Noch offen", open_count)
    c3.metric("Zurückgegeben", returned_count)

    if not items:
        st.info("Noch keine Fundstücke vorhanden. Lege im Tab „Fundstück melden“ den ersten Eintrag an.")
    else:
        df = pd.DataFrame(
            [{k: v for k, v in item.items() if k != "image_bytes"} for item in items]
        )
        st.dataframe(df[["created_at", "title", "location", "status", "description"]], use_container_width=True, hide_index=True)

with tab_add:
    st.subheader("Neues Fundstück erfassen")
    with st.form("add_item_form", clear_on_submit=False):
        title = st.text_input("Bezeichnung *", key="title", placeholder="z. B. Schwarzer Rucksack")
        description = st.text_area("Beschreibung", key="description", placeholder="Besondere Merkmale, Inhalt, Zustand …")
        location = st.text_input("Fundort *", key="location", placeholder="z. B. Bibliothek, Raum 204")
        contact = st.text_input("Kontakt / Abholhinweis (optional)", key="contact")
        uploaded = st.file_uploader("Foto (optional)", type=["png", "jpg", "jpeg", "webp"])
        use_ai = st.checkbox("Foto mit Hugging Face analysieren (Token erforderlich)", value=False)
        submitted = st.form_submit_button("Fundstück speichern", type="primary")

    if submitted:
        if not title.strip() or not location.strip():
            st.error("Bitte mindestens Bezeichnung und Fundort ausfüllen.")
        else:
            ai_description = ""
            image_bytes = None
            image_mime = None
            if uploaded is not None:
                try:
                    image = Image.open(uploaded).convert("RGB")
                    preview = io.BytesIO()
                    image.save(preview, format="JPEG", quality=85)
                    image_bytes = preview.getvalue()
                    image_mime = "image/jpeg"
                    if use_ai:
                        with st.spinner("Foto wird analysiert …"):
                            ai_description, ai_error = classify_image(uploaded, st.session_state.hf_token)
                        if ai_error:
                            st.warning(ai_error)
                        elif ai_description:
                            st.success("KI-Analyse abgeschlossen.")
                except Exception as exc:
                    st.error(f"Das Foto konnte nicht gelesen werden: {exc}")
            elif use_ai:
                st.warning("Bitte lade ein Foto hoch, um die KI-Analyse zu verwenden.")

            item = {
                "id": str(uuid.uuid4())[:8],
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "title": title.strip(),
                "description": description.strip(),
                "location": location.strip(),
                "contact": contact.strip(),
                "status": "Offen",
                "ai_analysis": ai_description or "",
                "image_bytes": image_bytes,
                "image_mime": image_mime,
            }
            st.session_state.items.insert(0, item)
            st.success("Fundstück wurde in dieser Sitzung gespeichert.")
            if ai_description:
                st.markdown("**KI-Ergebnis**")
                st.write(ai_description)
            st.caption("Hinweis: Ohne gemeinsamen Speicher ist dieser Eintrag nicht für andere Nutzer sichtbar.")
            st.rerun()

with tab_search:
    st.subheader("Fundstücke suchen und verwalten")
    q = st.text_input("Suchbegriff", placeholder="Bezeichnung, Beschreibung oder Fundort")
    status_filter = st.selectbox("Status", ["Alle", "Offen", "Zurückgegeben"])
    filtered = []
    for item in st.session_state.items:
        searchable = " ".join([
            item["title"], item["description"], item["location"], item["contact"], item["ai_analysis"]
        ]).casefold()
        if q.casefold() in searchable and (status_filter == "Alle" or item["status"] == status_filter):
            filtered.append(item)

    if not filtered:
        st.info("Keine passenden Fundstücke gefunden.")
    else:
        for item in filtered:
            with st.expander(f'{item["title"]} · {item["location"]} · {item["status"]}'):
                if item.get("image_bytes"):
                    st.image(item["image_bytes"], width=280)
                st.write(f'**Fundort:** {item["location"]}')
                st.write(f'**Gemeldet:** {item["created_at"]}')
                st.write(f'**Beschreibung:** {item["description"] or "Keine Beschreibung"}')
                if item["contact"]:
                    st.write(f'**Kontakt / Hinweis:** {item["contact"]}')
                if item["ai_analysis"]:
                    st.write("**KI-Analyse:**")
                    st.write(item["ai_analysis"])
                col_status, col_delete = st.columns(2)
                new_status = col_status.selectbox(
                    "Status ändern",
                    ["Offen", "Zurückgegeben"],
                    index=0 if item["status"] == "Offen" else 1,
                    key=f'status_{item["id"]}',
                )
                if col_status.button("Status übernehmen", key=f'update_{item["id"]}'):
                    for stored in st.session_state.items:
                        if stored["id"] == item["id"]:
                            stored["status"] = new_status
                            break
                    st.rerun()
                if col_delete.button("Eintrag löschen", key=f'delete_{item["id"]}'):
                    st.session_state.items = [
                        stored for stored in st.session_state.items if stored["id"] != item["id"]
                    ]
                    st.rerun()

with tab_export:
    st.subheader("Daten exportieren")
    if not st.session_state.items:
        st.info("Noch keine Daten zum Export vorhanden.")
    else:
        export_rows = []
        for item in st.session_state.items:
            export_rows.append({
                "ID": item["id"],
                "Erstellt am": item["created_at"],
                "Bezeichnung": item["title"],
                "Beschreibung": item["description"],
                "Fundort": item["location"],
                "Kontakt / Hinweis": item["contact"],
                "Status": item["status"],
                "KI-Analyse": item["ai_analysis"],
            })
        export_df = pd.DataFrame(export_rows)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "CSV herunterladen",
            data=csv_bytes,
            file_name="fundfinder_export.csv",
            mime="text/csv",
            type="primary",
        )
        st.caption("Fotos sind nicht in der CSV enthalten.")
