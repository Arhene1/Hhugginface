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
