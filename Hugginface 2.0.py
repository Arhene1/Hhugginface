
import os
import base64
import json
from openai import OpenAI

def classify_image(image_bytes):
    token = os.getenv("HF_TOKEN")

    if not token:
        try:
            token = st.secrets["HF_TOKEN"]
        except Exception:
            raise ValueError(
                "HF_TOKEN fehlt. Bitte den Token in den "
                "Umgebungsvariablen oder Streamlit Secrets hinterlegen."
            )

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = """
    Analysiere dieses Foto für ein deutsches Fundbüro.

    Bestimme den sichtbaren Gegenstand und wähle genau eine Kategorie:
    Schlüssel, Tasche / Rucksack, Geldbörse, Kleidung, Elektronik,
    Dokumente / Ausweise, Schmuck / Uhr, Brille, Schirm,
    Flasche / Trinkgefäß, Spielzeug, Sonstiges.

    Erfinde keine Merkmale, die nicht sichtbar sind.
    Wenn der Gegenstand nicht eindeutig erkennbar ist,
    verwende Sonstiges.

    Antworte ausschließlich als JSON mit diesen Feldern:
    {
      "gegenstand": "kurze Bezeichnung",
      "kategorie": "eine Kategorie aus der Liste",
      "beschreibung": "sichtbare Merkmale",
      "sicherheit": 0.0
    }

    Sicherheit ist eine geschätzte Zahl zwischen 0 und 1.
    """

    response = client.chat.completions.create(
        model=os.getenv(
            "HF_IMAGE_MODEL",
            "Qwen/Qwen2.5-VL-3B-Instruct",
        ),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/jpeg;base64,"
                                + image_b64
                            )
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    result = response.choices[0].message.content

    if not result:
        raise ValueError("Die KI hat keine Antwort geliefert.")

    # Entfernt optionale Markdown-Codeblöcke
    result = result.strip()
    if result.startswith("```"):
        result = result.split("\n", 1)[1]
        result = result.rsplit("```", 1)[0].strip()

    data = json.loads(result)

    if data.get("kategorie") not in CATEGORIES:
        data["kategorie"] = "Sonstiges"

    return data
