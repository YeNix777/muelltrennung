import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import os

# --- 1. SEITEN-LAYOUT ---
st.set_page_config(page_title="KI Müll-Scanner", page_icon="♻️", layout="centered")

st.title("♻️ Der smarte KI-Müll-Scanner")
st.write(
    "Projekt-Unterstützung für **SDG 12**. "
    "Mach ein Foto oder lade ein Bild hoch, um zu sehen, in welche Tonne es gehört."
)

MODEL_PATH = "muell_scanner_model_efficientnet.pth"

# --- INTERNER DIAGNOSE-CHECK ---
st.sidebar.header("⚙️ System-Status")

if os.path.exists(MODEL_PATH):
    st.sidebar.success(f"Datei '{MODEL_PATH}' wurde im Ordner gefunden!")
else:
    st.sidebar.error(f"⚠️ Datei '{MODEL_PATH}' FEHLT im aktuellen Ordner!")

# --- 2. SICHERES MODELL-LADEN ---
@st.cache_resource
def load_trained_model():
    # Architektur exakt wie beim Training definieren
    model = models.efficientnet_b0()
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 4)

    # Sicherer Import auf CPU
    state_dict = torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu"),
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()
    return model

# Modell laden mit sichtbarem Fehler-Reporting
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = load_trained_model()
        st.sidebar.success("KI-Modell erfolgreich geladen!")
    else:
        st.warning(
            "Bitte lege deine '.pth'-Datei in den Projektordner, "
            "um den Scanner freizuschalten."
        )
except Exception as e:
    st.error(
        "❗ Fehler beim Laden des Modells. Das bedeutet meistens, dass der "
        "Dateiname stimmt, das Modell in Colab aber mit einer anderen Architektur "
        "(z.B. MobileNet statt EfficientNet) trainiert wurde."
    )
    st.exception(e)

# --- 3. KLASSEN-MAPPING ---
klassen_mapping = {
    0: {
        "name": "Sondermüll / Gefahrgut ⚠️",
        "tipp": "Wertstoffhof / Schadstoffsammelstelle."
    },
    1: {
        "name": "Restmüll 🟤",
        "tipp": "Windeln, Plastikfolien, Hygienetücher, Keramik. Wird verbrannt."
    },
    2: {
        "name": "Biomüll 🟢",
        "tipp": "Grüne/braune Biotonne oder Kompost."
    },
    3: {
        "name": "Recyclebar ♻️",
        "tipp": "Plastikflaschen, Dosen, Glas, Pappe und Papier."
    }
}

# --- 4. OBERFLÄCHE ---
if model is not None:
    input_method = st.radio(
        "Bildquelle wählen:",
        ["Kamera", "Datei hochladen"],
        horizontal=True
    )

    if input_method == "Kamera":
        image_file = st.camera_input("Mach ein Foto vom Müllobjekt")
    else:
        image_file = st.file_uploader(
            "Wähle ein Müll-Foto aus:",
            type=["jpg", "jpeg", "png"]
        )

    if image_file is not None:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Ausgewähltes Foto", use_container_width=True)

        with st.spinner("KI analysiert..."):
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ])

            image_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                conf, predicted = torch.max(probabilities, 1)

                klasse_id = predicted.item()
                sicherheit = conf.item() * 100

        ergebnis = klassen_mapping[klasse_id]

        st.markdown(f"### Ergebnis: **{ergebnis['name']}**")
        st.info(f"💡 **Entsorgung:** {ergebnis['tipp']}")
        st.write(f"Sicherheit: {sicherheit:.1f}%")
        st.progress(int(sicherheit))

        if sicherheit < 60:
            st.warning(
                "Die KI ist sich nicht sehr sicher. Bitte prüfe das Ergebnis "
                "oder mache ein neues Foto mit besserem Licht."
            )
else:
    st.info("Die App wartet auf ein geladenes KI-Modell.")
