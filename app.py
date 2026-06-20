import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import os
import math
import pandas as pd

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
    model = models.efficientnet_b0()
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 4)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu"),
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()
    return model

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
        "tipp": "Wertstoffhof / Schadstoffsammelstelle.",
        "station_type": "Recyclinghof"
    },
    1: {
        "name": "Restmüll 🟤",
        "tipp": "Graue Restmülltonne. Größere Mengen zum Recyclinghof.",
        "station_type": "Recyclinghof"
    },
    2: {
        "name": "Biomüll 🟢",
        "tipp": "Grüne/braune Biotonne oder Kompost.",
        "station_type": "Biotonne"
    },
    3: {
        "name": "Recyclebar ♻️",
        "tipp": "Plastikflaschen, Dosen, Glas, Pappe und Papier.",
        "station_type": "Recyclinghof"
    }
}

# --- 4. DEMO-STANDORTE ---
district_points = {
    "Innenstadt / Dellviertel": (51.4344, 6.7623),
    "Duisburg-Nord": (51.4938, 6.7602),
    "Duisburg-Süd": (51.3658, 6.7594),
    "Rheinhausen": (51.4017, 6.7067),
    "Homberg / Ruhrort": (51.4538, 6.7130),
    "Meiderich / Beeck": (51.4655, 6.7756),
}

recycling_stations = [
    {
        "name": "Recyclinghof Nord (Demo)",
        "lat": 51.4890,
        "lon": 6.7630,
        "accepts": ["Recyclinghof"],
        "note": "Für Sondermüll, Elektrogeräte und größere Wertstoffe geeignet."
    },
    {
        "name": "Recyclinghof Mitte (Demo)",
        "lat": 51.4327,
        "lon": 6.7620,
        "accepts": ["Recyclinghof"],
        "note": "Zentraler Sammelpunkt für schwer einzuordnende Abfälle."
    },
    {
        "name": "Recyclinghof Süd (Demo)",
        "lat": 51.3588,
        "lon": 6.7482,
        "accepts": ["Recyclinghof"],
        "note": "Sammelpunkt für südliche Stadtteile."
    },
    {
        "name": "Bioabfall-Sammelpunkt Innenstadt (Demo)",
        "lat": 51.4372,
        "lon": 6.7714,
        "accepts": ["Biotonne"],
        "note": "Für Bioabfall im Alltag reicht normalerweise die eigene Biotonne."
    },
]


def distance_km(start, station):
    lat1, lon1 = map(math.radians, start)
    lat2, lon2 = math.radians(station["lat"]), math.radians(station["lon"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )

    return 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def show_nearest_station(station_type):
    st.markdown("### 📍 Nächste passende Sammelstelle")
    st.caption(
        "Demo-Funktion: Für einen echten Einsatz sollten hier offizielle "
        "Standorte und Öffnungszeiten der Stadt eingebunden werden."
    )

    district = st.selectbox(
        "Wo bist du ungefähr?",
        list(district_points.keys())
    )

    user_location = district_points[district]

    matching_stations = [
        station for station in recycling_stations
        if station_type in station["accepts"]
    ]

    if not matching_stations:
        st.info("Für diese Kategorie ist normalerweise keine Sammelstelle nötig.")
        return

    nearest = min(
        matching_stations,
        key=lambda station: distance_km(user_location, station)
    )

    dist = distance_km(user_location, nearest)

    st.success(f"Nächster Punkt: {nearest['name']} ({dist:.1f} km Luftlinie)")
    st.write(nearest["note"])

    map_data = pd.DataFrame([
        {"lat": user_location[0], "lon": user_location[1]},
        {"lat": nearest["lat"], "lon": nearest["lon"]},
    ])

    st.map(map_data, latitude="lat", longitude="lon", size=80)

    google_maps_url = (
        "https://www.google.com/maps/search/?api=1&query="
        + nearest["name"].replace(" ", "+")
        + "+Duisburg"
    )

    st.link_button("Route / Standort in Google Maps öffnen", google_maps_url)


# --- 5. OBERFLÄCHE ---
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

        show_nearest_station(ergebnis["station_type"])

else:
    st.info("Die App wartet auf ein geladenes KI-Modell.")
