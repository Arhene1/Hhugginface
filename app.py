import os
from datetime import datetime

import streamlit as st
from huggingface_hub import InferenceClient

# ------------------------------------------------------------
# Streamlit configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Hugging Face AI Studio",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SYSTEM_PROMPT = (
    "Du bist ein hilfreicher, präziser KI-Assistent. "
    "Antworte standardmäßig auf Deutsch. Wenn du etwas nicht sicher weißt, "
    "sage es offen und erfinde keine Fakten."
)

# ------------------------------------------------------------
# Minimal custom styling for a dashboard-like interface
# ------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.22); }
      .hero {
        padding: 1.25rem 1.4rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 18px;
        margin-bottom: 1rem;
        background: linear-gradient(120deg, rgba(99,102,241,.12), rgba(20,184,166,.08));
      }
      .hero h1 { margin: 0 0 .35rem 0; font-size: 1.8rem; }
      .hero p { margin: 0; opacity: .78; }
      div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.22);
        padding: .85rem 1rem;
        border-radius: 14px;
      }
      .small-muted { opacity: .7; font-size: .88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hallo! Ich bin dein KI-Assistent. Stelle mir eine Frage, oder wähle unten einen Beispiel-Prompt.",
            "time": datetime.now().strftime("%H:%M"),
        }
    ]
if "request_count" not in st.session_state:
    st.session_state.request_count = 0
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT


def reset_chat():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Neuer Chat gestartet. Wie kann ich dir helfen?",
            "time": datetime.now().strftime("%H:%M"),
        }
    ]


# ------------------------------------------------------------
# Sidebar settings
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🤗 AI Studio")
    st.caption("Streamlit · Hugging Face Inference API")
    st.divider()

    st.markdown("### Verbindung")
    token_from_env = os.getenv("HF_TOKEN", "")
    token_input = st.text_input(
        "Hugging Face Access Token",
        value=token_from_env,
        type="password",
        help="Wird nur für diese Sitzung verwendet. Für Deployments empfiehlt sich ein Secret namens HF_TOKEN.",
        placeholder="hf_…",
    )
    model_id = st.text_input(
        "Modell-ID",
        value=DEFAULT_MODEL,
        help="Verwende ein Chat-Modell, das über einen Hugging Face Inference Provider verfügbar ist.",
    )

    st.markdown("### Antwortverhalten")
    temperature = st.slider("Kreativität", min_value=0.0, max_value=1.5, value=0.7, step=0.1)
    max_tokens = st.slider("Max. Antwort-Tokens", min_value=128, max_value=2048, value=700, step=64)
    history_limit = st.slider(
        "Kontext: letzte Nachrichtenpaare",
        min_value=1,
        max_value=20,
        value=8,
        help="Begrenzt den Chatverlauf, der an die API gesendet wird.",
    )
    system_prompt = st.text_area(
        "Systemanweisung",
        value=st.session_state.system_prompt,
        height=130,
        help="Legt Rolle, Sprache und Antwortstil der KI fest.",
    )
    st.session_state.system_prompt = system_prompt

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧹 Chat leeren", use_container_width=True):
            reset_chat()
            st.rerun()
    with col_b:
        if st.button("⬇️ Export", use_container_width=True):
            export_text = "\n\n".join(
                f"{m['role'].upper()} ({m.get('time', '')})\n{m['content']}"
                for m in st.session_state.messages
            )
            st.download_button(
                "Download TXT",
                data=export_text,
                file_name="ki-chat.txt",
                mime="text/plain",
                use_container_width=True,
            )

    st.markdown("---")
    st.markdown(
        '<p class="small-muted">Tipp: Speichere Tokens nicht direkt im Quellcode '
        'und veröffentliche sie niemals in Git-Repositories.</p>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# Main dashboard header and status metrics
# ------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>Dein persönliches KI-Dashboard</h1>
      <p>Chatte mit einem Modell von Hugging Face und passe Modell, Kontext und Antwortstil links an.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

user_messages = sum(1 for m in st.session_state.messages if m["role"] == "user")
assistant_messages = sum(1 for m in st.session_state.messages if m["role"] == "assistant")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Modell", model_id.split("/")[-1] if model_id.strip() else "Nicht gesetzt")
m2.metric("Deine Nachrichten", user_messages)
m3.metric("KI-Antworten", max(assistant_messages - 1, 0))
m4.metric("API-Aufrufe", st.session_state.request_count)

st.write("")

chat_col, info_col = st.columns([2.25, 1], gap="large")

with chat_col:
    st.subheader("💬 Chat")
    st.caption("Der Gesprächsverlauf bleibt in der aktuellen Streamlit-Sitzung erhalten.")

    # Render conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("time"):
                st.caption(message["time"])

    # Quick prompt buttons
    st.markdown("**Schnellstart**")
    q1, q2, q3 = st.columns(3)
    selected_prompt = None
    if q1.button("Python erklären", use_container_width=True):
        selected_prompt = "Erkläre mir Python für Anfänger mit einem kleinen Beispiel."
    if q2.button("Ideen sammeln", use_container_width=True):
        selected_prompt = "Gib mir fünf kreative Ideen für ein kleines Python-Projekt."
    if q3.button("Text verbessern", use_container_width=True):
        selected_prompt = "Gib mir eine kurze Checkliste, mit der ich einen Text klarer und verständlicher mache."

    prompt = st.chat_input("Schreibe deine Nachricht …")
    if selected_prompt and not prompt:
        prompt = selected_prompt

    if prompt:
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "time": datetime.now().strftime("%H:%M")}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        if not token_input.strip():
            st.error(
                "Bitte gib links einen Hugging Face Access Token ein oder setze die Umgebungsvariable HF_TOKEN."
            )
        elif not model_id.strip():
            st.error("Bitte gib links eine Modell-ID ein.")
        else:
            with st.chat_message("assistant"):
                answer_placeholder = st.empty()
                try:
                    client = InferenceClient(
                        model=model_id.strip(),
                        token=token_input.strip(),
                    )

                    # Include system prompt plus a bounded recent conversation.
                    recent_messages = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                        if m["role"] in ("user", "assistant")
                    ]
                    # Keep the most recent N user/assistant messages; the system message is added separately.
                    recent_messages = recent_messages[-(history_limit * 2):]
                    api_messages = []
                    if system_prompt.strip():
                        api_messages.append({"role": "system", "content": system_prompt.strip()})
                    api_messages.extend(recent_messages)

                    answer_parts = []
                    stream = client.chat_completion(
                        messages=api_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                    )
                    for chunk in stream:
                        if not chunk.choices:
                            continue
                        piece = chunk.choices[0].delta.content or ""
                        if piece:
                            answer_parts.append(piece)
                            answer_placeholder.markdown("".join(answer_parts) + "▌")

                    answer = "".join(answer_parts).strip()
                    if not answer:
                        answer = "Das Modell hat keine Textantwort zurückgegeben."
                    answer_placeholder.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "time": datetime.now().strftime("%H:%M")}
                    )
                    st.session_state.request_count += 1
                    st.rerun()

                except Exception as exc:
                    answer_placeholder.empty()
                    st.error(
                        "Die Anfrage an Hugging Face ist fehlgeschlagen. Prüfe den Token, die Modell-ID, "
                        "die Provider-Verfügbarkeit und deinen API-Zugang."
                    )
                    with st.expander("Technische Details"):
                        st.code(str(exc))

with info_col:
    st.subheader("⚙️ Übersicht")
    st.info(
        "Diese App nutzt die Hugging Face Inference API. Das Modell läuft bei einem unterstützten "
        "Inference Provider, nicht automatisch lokal auf deinem Rechner."
    )
    st.markdown("**Aktuelle Konfiguration**")
    st.write(f"**Modell:** `{model_id}`")
    st.write(f"**Kreativität:** `{temperature}`")
    st.write(f"**Max. Tokens:** `{max_tokens}`")
    st.write(f"**Kontext:** letzte {history_limit} Nachrichtenpaare")
    st.write(f"**Token konfiguriert:** {'Ja' if token_input.strip() else 'Nein'}")

    st.markdown("**Nützliche Links**")
    st.markdown("- [Hugging Face Tokens](https://huggingface.co/settings/tokens)")
    st.markdown("- [Modell-Hub](https://huggingface.co/models)")
    st.markdown("- [InferenceClient-Dokumentation](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client)")

st.caption("Hinweis: Modellzugang, Verfügbarkeit und mögliche Kosten hängen von Hugging Face und dem gewählten Inference Provider ab.")
