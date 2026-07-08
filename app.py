from __future__ import annotations

import hashlib
import math
import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from PIL import Image

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models
    import torchvision.transforms as transforms
except ModuleNotFoundError as exc:
    torch = None
    nn = None
    F = None
    models = None
    transforms = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


MODEL_PATH = "muell_scanner_model_efficientnet.pth"
LEVEL_SIZE = 120


st.set_page_config(page_title="Sortify", page_icon="♻️", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #17352d;
        --muted: #607169;
        --line: #dfe7e1;
        --paper: #f6f4ec;
        --green: #1f7a5b;
    }
    .stApp {
        background:
            radial-gradient(circle at 82% 4%, rgba(31,122,91,.14), transparent 26rem),
            linear-gradient(180deg, #fbfaf5 0%, var(--paper) 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1120px; padding-top: 1.5rem; padding-bottom: 4rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    .hero {
        display: grid; grid-template-columns: 1.35fr .65fr; gap: 1rem; align-items: stretch;
        margin-bottom: 1rem;
    }
    .hero-main {
        position: relative; overflow: hidden;
        background: linear-gradient(135deg, #12382f, #1f7a5b);
        color: white; border-radius: 20px; padding: 1.6rem 1.7rem;
        box-shadow: 0 16px 32px rgba(24, 54, 45, .14);
    }
    .hero-main h1 { color: white; margin: .25rem 0 .5rem; font-size: 2.7rem; }
    .hero-main p { color: #e9f5ef; max-width: 680px; margin: 0; font-size: 1.05rem; }
    .kicker { color: #bce5d3; font-size: .76rem; font-weight: 850; letter-spacing: .14em; }
    .float-icons {
        display: flex; gap: .5rem; flex-wrap: wrap; margin-top: 1rem;
    }
    .float-icon {
        width: 42px; height: 42px; display: grid; place-items: center;
        border-radius: 14px; background: rgba(255,255,255,.14);
        border: 1px solid rgba(255,255,255,.16); font-size: 1.25rem;
        animation: bob 2.6s ease-in-out infinite;
    }
    .float-icon:nth-child(2) { animation-delay: .25s; }
    .float-icon:nth-child(3) { animation-delay: .5s; }
    .float-icon:nth-child(4) { animation-delay: .75s; }
    @keyframes bob {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        50% { transform: translateY(-5px) rotate(2deg); }
    }
    .mission-card, .panel {
        background: rgba(255,255,255,.9); border: 1px solid var(--line);
        border-radius: 16px; padding: 1rem; box-shadow: 0 8px 24px rgba(24,54,45,.06);
    }
    .mission-card { display: flex; flex-direction: column; justify-content: center; }
    .mission-card strong { color: var(--ink); font-size: 1.1rem; }
    .muted { color: var(--muted); font-size: .86rem; }
    .dashboard {
        display: grid; grid-template-columns: 1.1fr .9fr; gap: .85rem; margin: .9rem 0 1.2rem;
    }
    .stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .6rem; margin-top: .75rem; }
    .stat {
        background: #f5f7f4; border-radius: 12px; padding: .75rem; min-height: 76px;
        border: 1px solid #e7ece8;
    }
    .stat strong { display: block; color: var(--ink); font-size: 1.35rem; line-height: 1.15; }
    .stat span { color: var(--muted); font-size: .75rem; font-weight: 760; }
    .level-row, .challenge-row {
        display: flex; justify-content: space-between; align-items: center; gap: 1rem;
    }
    .level-row strong, .challenge-row strong { color: var(--ink); font-size: 1.12rem; }
    .bar { height: 10px; background: #e3e9e5; border-radius: 99px; overflow: hidden; margin-top: .55rem; }
    .bar span {
        display: block; height: 100%; background: linear-gradient(90deg, #1f7a5b, #55ba82);
        border-radius: 99px; animation: fillpop .7s ease-out both;
    }
    @keyframes fillpop {
        from { transform: scaleX(.15); transform-origin: left; }
        to { transform: scaleX(1); transform-origin: left; }
    }
    .badges { display: flex; flex-wrap: wrap; gap: .45rem; margin-top: .75rem; }
    .badge {
        display: inline-flex; align-items: center; border-radius: 99px;
        background: #e3f0e9; color: #245844; padding: .36rem .6rem;
        font-size: .76rem; font-weight: 800; animation: popin .35s ease-out both;
    }
    @keyframes popin {
        from { opacity: 0; transform: scale(.92); }
        to { opacity: 1; transform: scale(1); }
    }
    .reward {
        border: 1px solid #b8dfcb; background: #edf8f2; color: #214c3d;
        border-radius: 12px; padding: .75rem .85rem; margin: .8rem 0; font-weight: 800;
        animation: rewardPulse .8s ease-out both;
    }
    @keyframes rewardPulse {
        0% { transform: scale(.98); box-shadow: 0 0 0 rgba(31,122,91,0); }
        45% { transform: scale(1.015); box-shadow: 0 0 0 7px rgba(31,122,91,.12); }
        100% { transform: scale(1); box-shadow: 0 0 0 rgba(31,122,91,0); }
    }
    .result {
        background: white; border: 1px solid var(--line); border-radius: 16px;
        padding: 1rem; margin-top: .75rem;
    }
    .result-head {
        display: flex; justify-content: space-between; gap: 1rem; align-items: center;
        border-radius: 12px; color: white; padding: .9rem 1rem; margin-bottom: .8rem;
    }
    .result-head h2 { color: white; margin: 0; font-size: 1.35rem; }
    .result-icon {
        width: 58px; height: 58px; display: grid; place-items: center;
        border-radius: 18px; background: rgba(255,255,255,.18); font-size: 1.9rem;
        animation: bob 2.4s ease-in-out infinite;
    }
    .step {
        display: inline-flex; align-items: center; gap: .45rem;
        background: #e1eee8; color: #245844; padding: .42rem .7rem;
        border-radius: 999px; font-size: .78rem; font-weight: 800; margin: .7rem 0;
    }
    .scan-zone {
        border: 1px dashed #93b9a5; border-radius: 18px; padding: 1rem;
        background: linear-gradient(180deg, rgba(255,255,255,.86), rgba(238,248,242,.86));
    }
    .scanner-ring {
        width: 78px; height: 78px; margin: .2rem auto 1rem; border-radius: 50%;
        border: 7px solid #dfece5; border-top-color: #1f7a5b; border-right-color: #55ba82;
        animation: spin 1.6s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .category-row { display: flex; flex-wrap: wrap; gap: .5rem; margin: .75rem 0; }
    .category-chip {
        display: inline-flex; align-items: center; gap: .35rem; border-radius: 999px;
        background: #f5f7f4; border: 1px solid #e1e8e2; padding: .4rem .65rem;
        font-size: .78rem; font-weight: 800; color: var(--ink);
    }
    .guide-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .75rem; }
    .guide-card {
        background: white; border: 1px solid var(--line); border-radius: 14px;
        padding: .9rem; min-height: 142px;
        transition: transform .18s ease, box-shadow .18s ease;
    }
    .guide-card:hover { transform: translateY(-3px); box-shadow: 0 10px 20px rgba(24,54,45,.08); }
    .guide-card strong { color: var(--ink); display: block; margin-bottom: .35rem; }
    @media (max-width: 820px) {
        .hero, .dashboard, .guide-grid { grid-template-columns: 1fr; }
        .stat-grid { grid-template-columns: repeat(2, 1fr); }
        .hero-main h1 { font-size: 2.15rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


CLASS_MAPPING = {
    0: {
        "name": "Hazardous waste",
        "short": "Hazardous",
        "icon": "⚠️",
        "tip": "Take it to a recycling center or hazardous waste collection point.",
        "station_type": "Recycling Center",
        "color": "#c65345",
        "points": 60,
        "impact": 1.0,
    },
    1: {
        "name": "Residual waste",
        "short": "Residual",
        "icon": "🗑️",
        "tip": "Use the grey residual waste bin. Larger amounts should go to a recycling center.",
        "station_type": "Recycling Center",
        "color": "#59636e",
        "points": 30,
        "impact": 0.05,
    },
    2: {
        "name": "Organic waste",
        "short": "Organic",
        "icon": "🍃",
        "tip": "Use the green or brown organic waste bin, or compost where allowed.",
        "station_type": "Organic Waste Bin",
        "color": "#4f9b70",
        "points": 40,
        "impact": 0.12,
    },
    3: {
        "name": "Recyclable",
        "short": "Recyclable",
        "icon": "♻️",
        "tip": "Plastic bottles, cans, glass, cardboard, and paper belong in suitable recycling streams.",
        "station_type": "Recycling Center",
        "color": "#366fae",
        "points": 45,
        "impact": 0.35,
    },
}


DISTRICT_POINTS = {
    "City Center / Dellviertel": (51.4344, 6.7623),
    "Duisburg North": (51.4938, 6.7602),
    "Duisburg South": (51.3658, 6.7594),
    "Rheinhausen": (51.4017, 6.7067),
    "Homberg / Ruhrort": (51.4538, 6.7130),
    "Meiderich / Beeck": (51.4655, 6.7756),
}


RECYCLING_STATIONS = [
    {
        "name": "Recycling Center North (Demo)",
        "lat": 51.4890,
        "lon": 6.7630,
        "accepts": ["Recycling Center"],
        "note": "Suitable for hazardous waste, electronic devices, and larger recyclable items.",
    },
    {
        "name": "Recycling Center Central (Demo)",
        "lat": 51.4327,
        "lon": 6.7620,
        "accepts": ["Recycling Center"],
        "note": "Central collection point for waste that is difficult to classify.",
    },
    {
        "name": "Recycling Center South (Demo)",
        "lat": 51.3588,
        "lon": 6.7482,
        "accepts": ["Recycling Center"],
        "note": "Collection point for southern districts.",
    },
    {
        "name": "Organic Waste Collection Point City Center (Demo)",
        "lat": 51.4372,
        "lon": 6.7714,
        "accepts": ["Organic Waste Bin"],
        "note": "For everyday organic waste, the household organic bin is usually enough.",
    },
]


@st.cache_resource(show_spinner=False)
def load_trained_model():
    if TORCH_IMPORT_ERROR is not None:
        raise RuntimeError("PyTorch and torchvision are required to load the scanner.") from TORCH_IMPORT_ERROR

    model = models.efficientnet_b0()
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 4)
    state_dict = torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def default_player() -> dict:
    return {
        "points": 0,
        "scans": 0,
        "streak": 0,
        "last_scan_date": "",
        "categories": [],
        "hazardous_saves": 0,
        "impact": 0.0,
        "awarded_images": [],
    }


def player_state() -> dict:
    if "player" not in st.session_state:
        st.session_state.player = default_player()
    return st.session_state.player


def level_for_points(points: int) -> int:
    return points // LEVEL_SIZE + 1


def progress_percent(points: int) -> int:
    return int((points % LEVEL_SIZE) / LEVEL_SIZE * 100)


def update_streak(player: dict) -> None:
    today = date.today()
    today_key = today.isoformat()
    yesterday_key = (today - timedelta(days=1)).isoformat()
    last_scan = player.get("last_scan_date", "")

    if last_scan == today_key:
        player["streak"] = max(1, int(player.get("streak", 0)))
    elif last_scan == yesterday_key:
        player["streak"] = int(player.get("streak", 0)) + 1
    else:
        player["streak"] = 1

    player["last_scan_date"] = today_key


def badges_for_player(player: dict) -> list[str]:
    badges = []
    if player.get("scans", 0) >= 1:
        badges.append("🔍 First scan")
    if player.get("streak", 0) >= 3:
        badges.append("🔥 3 day streak")
    if len(player.get("categories", [])) >= 3:
        badges.append("🧭 Sorting explorer")
    if player.get("hazardous_saves", 0) >= 1:
        badges.append("🛡️ Safety hero")
    if player.get("points", 0) >= LEVEL_SIZE:
        badges.append("⭐ Level climber")
    return badges


def award_points(class_id: int, confidence: float, image_key: str) -> dict | None:
    player = player_state()
    if image_key in player.get("awarded_images", []):
        return None

    result = CLASS_MAPPING[class_id]
    known_categories = set(player.get("categories", []))
    new_category = result["short"] not in known_categories
    points = int(result["points"] + confidence // 10)

    if new_category:
        points += 15
    if class_id == 0:
        points += 20
        player["hazardous_saves"] = int(player.get("hazardous_saves", 0)) + 1

    player["points"] = int(player.get("points", 0)) + points
    player["scans"] = int(player.get("scans", 0)) + 1
    player["impact"] = round(float(player.get("impact", 0.0)) + result["impact"], 2)
    known_categories.add(result["short"])
    player["categories"] = sorted(known_categories)
    player.setdefault("awarded_images", []).append(image_key)
    update_streak(player)

    return {"points": points, "new_category": new_category}


def image_fingerprint(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()[:16]


def predict_image(image: Image.Image, model) -> tuple[int, float]:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        conf, predicted = torch.max(probabilities, 1)
    return predicted.item(), conf.item() * 100


def distance_km(start: tuple[float, float], station: dict) -> float:
    lat1, lon1 = map(math.radians, start)
    lat2, lon2 = math.radians(station["lat"]), math.radians(station["lon"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero">
            <div class="hero-main">
                <div class="kicker">SDG 12 WASTE SORTING APP</div>
                <h1>Sortify</h1>
                <p>Scan waste, learn where it belongs, and level up by making better
                disposal decisions around Duisburg.</p>
                <div class="float-icons">
                    <span class="float-icon">♻️</span>
                    <span class="float-icon">🍃</span>
                    <span class="float-icon">⚠️</span>
                    <span class="float-icon">🗑️</span>
                </div>
            </div>
            <div class="mission-card">
                <span class="muted">Today mission</span>
                <strong>🎯 Identify 3 different waste categories</strong>
                <span class="muted">Earn bonus progress by scanning varied items.</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard() -> None:
    player = player_state()
    points = int(player.get("points", 0))
    categories = player.get("categories", [])
    badges = badges_for_player(player)
    badge_html = "".join(
        f'<span class="badge">{badge}</span>' for badge in badges
    ) or '<span class="badge">🚀 Ready to start</span>'
    challenge_progress = min(100, int(len(categories) / 3 * 100))
    next_level = LEVEL_SIZE - (points % LEVEL_SIZE)

    st.markdown(
        f"""
        <section class="dashboard">
            <div class="panel">
                <div class="level-row">
                    <strong>Level {level_for_points(points)}</strong>
                    <span class="muted">{points} points</span>
                </div>
                <div class="bar"><span style="width:{progress_percent(points)}%"></span></div>
                <div class="stat-grid">
                    <div class="stat"><strong>🔍 {player.get("scans", 0)}</strong><span>Scans</span></div>
                    <div class="stat"><strong>🔥 {player.get("streak", 0)}</strong><span>Day streak</span></div>
                    <div class="stat"><strong>🏷️ {len(categories)}</strong><span>Categories</span></div>
                    <div class="stat"><strong>🌍 {player.get("impact", 0.0):.1f} kg</strong><span>CO2e saved</span></div>
                </div>
                <div class="badges">{badge_html}</div>
            </div>
            <div class="panel">
                <div class="challenge-row">
                    <div>
                        <strong>🎮 Challenge progress</strong><br>
                        <span class="muted">{len(categories)} of 3 categories discovered</span>
                    </div>
                    <strong>{challenge_progress}%</strong>
                </div>
                <div class="bar"><span style="width:{challenge_progress}%"></span></div>
                <p class="muted" style="margin:.8rem 0 0">{next_level} points to the next level.</p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def show_nearest_station(station_type: str) -> None:
    st.markdown('<div class="step">Nearest suitable collection point</div>', unsafe_allow_html=True)
    st.caption(
        "Demo feature: In a real deployment, official city locations and opening hours should be integrated here."
    )

    district = st.selectbox("Where are you approximately located?", list(DISTRICT_POINTS))
    user_location = DISTRICT_POINTS[district]
    matching_stations = [
        station for station in RECYCLING_STATIONS if station_type in station["accepts"]
    ]

    if not matching_stations:
        st.info("For this category, a collection point is usually not required.")
        return

    nearest = min(matching_stations, key=lambda station: distance_km(user_location, station))
    dist = distance_km(user_location, nearest)

    st.success(f"Nearest point: {nearest['name']} ({dist:.1f} km straight-line distance)")
    st.write(nearest["note"])
    st.map(
        pd.DataFrame(
            [
                {"lat": user_location[0], "lon": user_location[1]},
                {"lat": nearest["lat"], "lon": nearest["lon"]},
            ]
        ),
        latitude="lat",
        longitude="lon",
        size=80,
    )

    google_maps_url = (
        "https://www.google.com/maps/search/?api=1&query="
        + nearest["name"].replace(" ", "+")
        + "+Duisburg"
    )
    st.link_button("Open route / location in Google Maps", google_maps_url)


def render_result(class_id: int, confidence: float, reward: dict | None) -> None:
    result = CLASS_MAPPING[class_id]
    if reward:
        extra = " New category discovered." if reward["new_category"] else ""
        st.markdown(
            f'<div class="reward">+{reward["points"]} points earned.{extra}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="result">
            <div class="result-head" style="background:{result["color"]}">
                <div>
                    <span style="font-size:.75rem;font-weight:850;text-transform:uppercase;opacity:.82">Result</span>
                    <h2>{result["name"]}</h2>
                </div>
                <div class="result-icon">{result["icon"]}</div>
            </div>
            <div class="category-row">
                <span class="category-chip">🎯 {confidence:.1f}% confidence</span>
                <span class="category-chip">⭐ +{result["points"]} base points</span>
                <span class="category-chip">🌍 {result["impact"]:.2f} kg impact</span>
            </div>
            <p><strong>Disposal guidance:</strong><br>{result["tip"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(int(confidence))
    if confidence < 60:
        st.warning(
            "The AI is not very confident. Please check the result or take a new photo with better lighting."
        )
    show_nearest_station(result["station_type"])


def render_scanner(model) -> None:
    st.markdown('<div class="step">Scan mission</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="scan-zone">
            <div class="scanner-ring"></div>
            <strong>Scanner ready</strong><br>
            <span class="muted">Add one clear item, then launch the scan to earn points.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    input_method = st.radio("Choose image source:", ["Camera", "Upload file"], horizontal=True)

    if input_method == "Camera":
        image_file = st.camera_input("Take a photo of the waste item")
    else:
        image_file = st.file_uploader("Choose a waste photo:", type=["jpg", "jpeg", "png"])

    if image_file is None:
        st.info("Add a clear photo of one waste item to start a scan.")
        return

    image_bytes = image_file.getvalue()
    image_key = image_fingerprint(image_bytes)
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Selected photo", use_container_width=True)

    if st.button("Analyze and earn points", type="primary", use_container_width=True):
        with st.spinner("AI is analyzing the item..."):
            class_id, confidence = predict_image(image, model)
        reward = award_points(class_id, confidence, image_key)
        st.session_state.last_prediction = {
            "image_key": image_key,
            "class_id": class_id,
            "confidence": confidence,
            "reward": reward,
        }

    prediction = st.session_state.get("last_prediction")
    if prediction and prediction["image_key"] == image_key:
        render_result(
            prediction["class_id"],
            prediction["confidence"],
            prediction.get("reward"),
        )


def render_demo_scanner(reason: str) -> None:
    st.markdown('<div class="step">Demo scan mission</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="scan-zone">
            <div class="scanner-ring"></div>
            <strong>AI scanner locked for now</strong><br>
            <span class="muted">{reason}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "The real photo scanner needs PyTorch and torchvision. Until those are installed, "
        "use this demo scan to test the game loop, rewards, result cards, quests, and badges."
    )
    demo_options = {
        f"{item['icon']} {item['name']}": class_id
        for class_id, item in CLASS_MAPPING.items()
    }
    selected = st.selectbox("Choose a demo item category:", list(demo_options))
    class_id = demo_options[selected]
    confidence = st.slider("Demo confidence", 45, 99, 86)

    if st.button("Run demo scan and earn points", type="primary", use_container_width=True):
        demo_key = f"demo_{class_id}_{confidence}_{date.today().isoformat()}"
        reward = award_points(class_id, float(confidence), demo_key)
        st.session_state.last_demo_prediction = {
            "class_id": class_id,
            "confidence": float(confidence),
            "reward": reward,
        }

    prediction = st.session_state.get("last_demo_prediction")
    if prediction:
        render_result(
            prediction["class_id"],
            prediction["confidence"],
            prediction.get("reward"),
        )


def render_quests() -> None:
    player = player_state()
    categories = set(player.get("categories", []))
    st.subheader("Quest log")
    quests = [
        ("First Scan", player.get("scans", 0) >= 1, "Scan one waste item."),
        ("Category Collector", len(categories) >= 3, "Discover three different categories."),
        ("Safety Check", player.get("hazardous_saves", 0) >= 1, "Correctly identify hazardous waste."),
        ("Consistency", player.get("streak", 0) >= 3, "Use Sortify three days in a row."),
    ]
    for title, done, text in quests:
        st.checkbox(f"{title}: {text}", value=done, disabled=True, key=f"quest_{title}")
        st.caption("Complete" if done else "Open")


def render_guide() -> None:
    st.subheader("Sorting guide")
    st.markdown('<div class="guide-grid">', unsafe_allow_html=True)
    for item in CLASS_MAPPING.values():
        st.markdown(
            f"""
            <div class="guide-card">
                <strong style="color:{item["color"]}">{item["icon"]} {item["name"]}</strong>
                <span class="muted">{item["tip"]}</span>
                <div class="category-row">
                    <span class="category-chip">⭐ {item["points"]} pts</span>
                    <span class="category-chip">🌍 {item["impact"]:.2f} kg</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


render_hero()

with st.sidebar:
    st.header("System status")
    if os.path.exists(MODEL_PATH):
        st.success("AI model file found.")
    else:
        st.error("AI model file is missing.")
    if st.button("Reset game progress"):
        st.session_state.player = default_player()
        st.session_state.pop("last_prediction", None)
        st.rerun()
    st.caption("Progress is stored only in this browser session.")

model = None
try:
    if TORCH_IMPORT_ERROR is not None:
        st.warning(
            "The real photo scanner is locked because PyTorch is not installed in this environment. "
            "You can still use the demo scanner below to test the game loop."
        )
        st.sidebar.warning("PyTorch is not installed.")
    elif os.path.exists(MODEL_PATH):
        model = load_trained_model()
        st.sidebar.success("AI model loaded.")
    else:
        st.warning("Place the .pth model file in the project folder to activate the scanner.")
except Exception as exc:
    st.error(
        "The model file was found, but it could not be loaded. Check that it was trained with the same EfficientNet architecture."
    )
    st.exception(exc)

render_dashboard()

scan_tab, quests_tab, guide_tab = st.tabs(["Scan", "Quests", "Guide"])

with scan_tab:
    if model is not None:
        render_scanner(model)
    elif TORCH_IMPORT_ERROR is not None:
        render_demo_scanner("Missing dependency: torch / torchvision.")
    else:
        render_demo_scanner("The model file is missing or could not be loaded.")

with quests_tab:
    render_quests()

with guide_tab:
    render_guide()
