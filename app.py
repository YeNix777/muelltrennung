import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import os
import math
import pandas as pd

# --- 1. PAGE LAYOUT ---
st.set_page_config(page_title="Sortify", page_icon="♻️", layout="centered")

st.title("♻️ Sortify")
st.write(
    "Project support for **SDG 12**. "
    "Take a photo or upload an image to see which waste category it belongs to."
)

MODEL_PATH = "muell_scanner_model_efficientnet.pth"

# --- INTERNAL DIAGNOSTIC CHECK ---
st.sidebar.header("⚙️ System Status")

if os.path.exists(MODEL_PATH):
    st.sidebar.success(f"File '{MODEL_PATH}' was found in the project folder!")
else:
    st.sidebar.error(f"⚠️ File '{MODEL_PATH}' is MISSING from the current folder!")

# --- 2. SAFE MODEL LOADING ---
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
        st.sidebar.success("AI model loaded successfully!")
    else:
        st.warning(
            "Please place your '.pth' model file in the project folder "
            "to activate the scanner."
        )
except Exception as e:
    st.error(
        "❗ Error while loading the model. This usually means that the filename "
        "is correct, but the model was trained in Colab with a different architecture "
        "(for example MobileNet instead of EfficientNet)."
    )
    st.exception(e)

# --- 3. CLASS MAPPING ---
class_mapping = {
    0: {
        "name": "Hazardous Waste ⚠️",
        "tip": "Take it to a recycling center or hazardous waste collection point.",
        "station_type": "Recycling Center"
    },
    1: {
        "name": "Residual Waste 🟤",
        "tip": "Use the grey residual waste bin. Larger amounts should go to a recycling center.",
        "station_type": "Recycling Center"
    },
    2: {
        "name": "Organic Waste 🟢",
        "tip": "Use the green/brown organic waste bin or compost.",
        "station_type": "Organic Waste Bin"
    },
    3: {
        "name": "Recyclable ♻️",
        "tip": "Plastic bottles, cans, glass, cardboard, and paper.",
        "station_type": "Recycling Center"
    }
}

# --- 4. DEMO LOCATIONS ---
district_points = {
    "City Center / Dellviertel": (51.4344, 6.7623),
    "Duisburg North": (51.4938, 6.7602),
    "Duisburg South": (51.3658, 6.7594),
    "Rheinhausen": (51.4017, 6.7067),
    "Homberg / Ruhrort": (51.4538, 6.7130),
    "Meiderich / Beeck": (51.4655, 6.7756),
}

recycling_stations = [
    {
        "name": "Recycling Center North (Demo)",
        "lat": 51.4890,
        "lon": 6.7630,
        "accepts": ["Recycling Center"],
        "note": "Suitable for hazardous waste, electronic devices, and larger recyclable items."
    },
    {
        "name": "Recycling Center Central (Demo)",
        "lat": 51.4327,
        "lon": 6.7620,
        "accepts": ["Recycling Center"],
        "note": "Central collection point for waste that is difficult to classify."
    },
    {
        "name": "Recycling Center South (Demo)",
        "lat": 51.3588,
        "lon": 6.7482,
        "accepts": ["Recycling Center"],
        "note": "Collection point for southern districts."
    },
    {
        "name": "Organic Waste Collection Point City Center (Demo)",
        "lat": 51.4372,
        "lon": 6.7714,
        "accepts": ["Organic Waste Bin"],
        "note": "For everyday organic waste, the household organic bin is usually enough."
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
    st.markdown("### 📍 Nearest Suitable Collection Point")
    st.caption(
        "Demo feature: In a real deployment, official locations and opening hours "
        "from the city should be integrated here."
    )

    district = st.selectbox(
        "Where are you approximately located?",
        list(district_points.keys())
    )

    user_location = district_points[district]

    matching_stations = [
        station for station in recycling_stations
        if station_type in station["accepts"]
    ]

    if not matching_stations:
        st.info("For this category, a collection point is usually not required.")
        return

    nearest = min(
        matching_stations,
        key=lambda station: distance_km(user_location, station)
    )

    dist = distance_km(user_location, nearest)

    st.success(f"Nearest point: {nearest['name']} ({dist:.1f} km straight-line distance)")
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

    st.link_button("Open route / location in Google Maps", google_maps_url)


# --- 5. USER INTERFACE ---
if model is not None:
    input_method = st.radio(
        "Choose image source:",
        ["Camera", "Upload file"],
        horizontal=True
    )

    if input_method == "Camera":
        image_file = st.camera_input("Take a photo of the waste item")
    else:
        image_file = st.file_uploader(
            "Choose a waste photo:",
            type=["jpg", "jpeg", "png"]
        )

    if image_file is not None:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Selected photo", use_container_width=True)

        with st.spinner("AI is analyzing..."):
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

                class_id = predicted.item()
                confidence = conf.item() * 100

        result = class_mapping[class_id]

        st.markdown(f"### Result: **{result['name']}**")
        st.info(f"💡 **Disposal guidance:** {result['tip']}")
        st.write(f"Confidence: {confidence:.1f}%")
        st.progress(int(confidence))

        if confidence < 60:
            st.warning(
                "The AI is not very confident. Please check the result "
                "or take a new photo with better lighting."
            )

        show_nearest_station(result["station_type"])

else:
    st.info("The app is waiting for a loaded AI model.")
