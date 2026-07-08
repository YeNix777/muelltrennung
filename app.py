from __future__ import annotations

import hashlib
import math
import os
from datetime import date, timedelta

import pandas as pd
import pydeck as pdk
import streamlit as st
import folium
from folium.plugins import LocateControl
from PIL import Image
from streamlit_folium import st_folium

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
        --muted: #3f554d;
        --line: #cbd9d1;
        --paper: #eef3ec;
        --green: #1f7a5b;
        --panel: #ffffff;
    }
    .stApp {
        background:
            radial-gradient(circle at 82% 4%, rgba(31,122,91,.16), transparent 24rem),
            linear-gradient(180deg, #f6faf5 0%, var(--paper) 100%);
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1120px; padding-top: 1.5rem; padding-bottom: 4rem; }
    h1, h2, h3, h4, h5, h6, p, li, label, span { color: var(--ink); letter-spacing: 0; }
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p {
        color: var(--ink);
    }
    [data-testid="stCaptionContainer"], .stCaptionContainer, small {
        color: var(--muted) !important;
    }
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
        background: var(--panel); border: 1px solid var(--line);
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
        background: #eef6f0; border-radius: 12px; padding: .75rem; min-height: 76px;
        border: 1px solid #d5e3d9;
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
        background: var(--panel); border: 1px solid var(--line); border-radius: 16px;
        padding: 1rem; margin-top: .75rem;
    }
    .result p, .result strong { color: var(--ink); }
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
        background: linear-gradient(180deg, #ffffff, #eaf5ee);
    }
    .scan-zone strong { color: var(--ink); }
    .scan-zone .muted { color: #52675f; }
    .scanner-ring {
        position: relative;
        width: 78px; height: 78px; margin: .2rem auto 1rem; border-radius: 50%;
        border: 7px solid #dfece5; border-top-color: #1f7a5b; border-right-color: #55ba82;
        animation: scanReady .65s ease-out both;
    }
    .scanner-ring::after {
        content: ""; position: absolute; inset: 17px; border-radius: 50%;
        background: #edf8f2; border: 1px solid #b8dfcb;
    }
    @keyframes scanReady {
        from { transform: scale(.94); opacity: .55; }
        to { transform: scale(1); opacity: 1; }
    }
    div[data-baseweb="tab-list"] { background: rgba(255,255,255,.76); border-radius: 14px; padding: .25rem; }
    div[data-baseweb="tab-list"] button p { color: var(--ink) !important; font-weight: 850; }
    div[data-baseweb="tab-list"] button[aria-selected="true"] p { color: #0f5f46 !important; }
    div[role="radiogroup"] label, div[role="radiogroup"] p, div[role="radiogroup"] span {
        color: var(--ink) !important;
    }
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
        color: var(--ink) !important; font-weight: 760;
    }
    [data-testid="stSelectbox"] label, [data-testid="stTextInput"] label,
    [data-testid="stTextArea"] label, [data-testid="stFileUploader"] label,
    [data-testid="stSlider"] label {
        color: var(--ink) !important;
    }
    input, textarea, [data-baseweb="select"] {
        color: var(--ink) !important;
        background-color: #ffffff !important;
    }
    [data-testid="stInfo"], [data-testid="stWarning"], [data-testid="stSuccess"], [data-testid="stAlert"] {
        color: var(--ink) !important;
        background: #ffffff !important;
        border: 1px solid var(--line) !important;
    }
    [data-testid="stInfo"] p, [data-testid="stWarning"] p, [data-testid="stSuccess"] p,
    [data-testid="stAlert"] p {
        color: var(--ink) !important;
    }
    .category-row { display: flex; flex-wrap: wrap; gap: .5rem; margin: .75rem 0; }
    .category-chip {
        display: inline-flex; align-items: center; gap: .35rem; border-radius: 999px;
        background: #eef6f0; border: 1px solid #d5e3d9; padding: .4rem .65rem;
        font-size: .78rem; font-weight: 800; color: var(--ink);
    }
    .guide-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .75rem; }
    .guide-card {
        background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
        padding: .9rem; min-height: 142px;
        transition: transform .18s ease, box-shadow .18s ease;
    }
    .guide-card:hover { transform: translateY(-3px); box-shadow: 0 10px 20px rgba(24,54,45,.08); }
    .guide-card strong { color: var(--ink); display: block; margin-bottom: .35rem; }
    .mission-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin: 1rem 0; }
    .mission-tile {
        background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
        padding: 1rem; min-height: 138px;
    }
    .mission-tile strong { color: var(--ink); display: block; margin-bottom: .4rem; }
    .live-progress {
        background: white; border: 1px solid var(--line); border-radius: 14px;
        padding: .9rem; margin: .9rem 0;
    }
    .map-list {
        display: grid; grid-template-columns: repeat(2, 1fr); gap: .65rem; margin-top: .8rem;
    }
    .map-item {
        background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
        padding: .8rem;
    }
    .leader-row {
        display: grid; grid-template-columns: 54px 1.4fr .7fr .7fr;
        gap: .75rem; align-items: center; background: var(--panel);
        border: 1px solid var(--line); border-radius: 12px; padding: .8rem;
        margin-bottom: .55rem;
    }
    .leader-row.is-player {
        border: 2px solid #1f7a5b; background: #edf8f2;
        box-shadow: 0 8px 20px rgba(31,122,91,.1);
    }
    .leader-rank { font-size: 1.25rem; font-weight: 900; text-align: center; }
    .leader-score { font-weight: 900; color: #155e47; }
    .leader-label { color: var(--muted); font-size: .72rem; font-weight: 750; }
    .profile-header {
        display: grid; grid-template-columns: .45fr 1.55fr; gap: 1rem; align-items: center;
        background: var(--panel); border: 1px solid var(--line); border-radius: 16px; padding: 1rem;
    }
    .avatar {
        width: 104px; height: 104px; border-radius: 30px; display: grid; place-items: center;
        background: linear-gradient(135deg, #1f7a5b, #55ba82); color: white; font-size: 2.4rem;
        animation: bob 2.8s ease-in-out infinite;
    }
    @media (max-width: 820px) {
        .hero, .dashboard, .guide-grid, .mission-grid, .map-list, .profile-header { grid-template-columns: 1fr; }
        .stat-grid { grid-template-columns: repeat(2, 1fr); }
        .hero-main h1 { font-size: 2.15rem; }
        .leader-row { grid-template-columns: 42px 1fr .65fr; }
        .leader-scans { display: none; }
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


COMMUNITY_NEIGHBORHOODS = {
    "Kibera, Nairobi": (-1.3133, 36.7892),
    "Mathare, Nairobi": (-1.2588, 36.8574),
    "Kayole, Nairobi": (-1.2795, 36.9120),
    "Dandora, Nairobi": (-1.2484, 36.8991),
    "Mukuru, Nairobi": (-1.3158, 36.8764),
}


DEFAULT_DROP_OFFS = [
    {
        "name": "Kibera community sorting hub",
        "lat": -1.3133,
        "lon": 36.7892,
        "accepts": ["Recycling Center", "Organic Waste Bin"],
        "note": "Demo community point for sorted recyclables and household organic waste.",
        "added_by": "Sortify",
    },
    {
        "name": "Mathare plastics collection point",
        "lat": -1.2588,
        "lon": 36.8574,
        "accepts": ["Recycling Center"],
        "note": "Demo point for bottles, cans, paper, and clean plastic packaging.",
        "added_by": "Sortify",
    },
    {
        "name": "Kayole safe disposal desk",
        "lat": -1.2795,
        "lon": 36.9120,
        "accepts": ["Recycling Center"],
        "note": "Demo point for items that need careful handling before collection.",
        "added_by": "Sortify",
    },
    {
        "name": "Mukuru organics exchange",
        "lat": -1.3158,
        "lon": 36.8764,
        "accepts": ["Organic Waste Bin"],
        "note": "Demo point for food scraps and other organic waste streams.",
        "added_by": "Sortify",
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


def drop_off_state() -> list[dict]:
    if "drop_offs" not in st.session_state:
        st.session_state.drop_offs = [dict(point) for point in DEFAULT_DROP_OFFS]
    return st.session_state.drop_offs


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
                <p>Scan waste, build better sorting habits, and help communities map
                useful drop-off areas in places where waste services are still developing.</p>
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


def render_live_points_progress(label: str = "Current progress") -> None:
    player = player_state()
    points = int(player.get("points", 0))
    st.markdown(
        f"""
        <div class="live-progress">
            <div class="level-row">
                <strong>{label}</strong>
                <span class="muted">Level {level_for_points(points)} · {points} points</span>
            </div>
            <div class="bar"><span style="width:{progress_percent(points)}%"></span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def map_dataframe(points: list[dict], preview: dict | None = None) -> pd.DataFrame:
    rows = []
    for point in points:
        rows.append(
            {
                "lat": point["lat"],
                "lon": point["lon"],
                "name": point["name"],
                "pin": "📍" if point.get("added_by") == "Community" else "♻️",
                "color": [34, 123, 88, 220],
                "size": 28 if point.get("added_by") == "Community" else 23,
            }
        )
    if preview:
        rows.append(
            {
                "lat": preview["lat"],
                "lon": preview["lon"],
                "name": preview["name"],
                "pin": "📌",
                "color": [210, 82, 67, 240],
                "size": 34,
            }
        )
    return pd.DataFrame(rows)


def render_pin_map(points: list[dict], preview: dict | None = None, zoom: float = 12.2) -> None:
    data = map_dataframe(points, preview)
    text_layer = pdk.Layer(
        "TextLayer",
        data,
        get_position="[lon, lat]",
        get_text="pin",
        get_size="size",
        get_color="color",
        get_alignment_baseline="'bottom'",
        pickable=True,
    )
    label_layer = pdk.Layer(
        "TextLayer",
        data,
        get_position="[lon, lat]",
        get_text="name",
        get_size=13,
        get_color=[23, 53, 45, 230],
        get_pixel_offset=[0, 14],
        pickable=True,
    )
    st.pydeck_chart(
        pdk.Deck(
            map_style="light",
            initial_view_state=pdk.ViewState(
                latitude=-1.2921,
                longitude=36.8219,
                zoom=zoom,
                pitch=0,
            ),
            layers=[text_layer, label_layer],
            tooltip={"text": "{name}"},
        ),
        use_container_width=True,
    )


def render_community_map() -> None:
    drop_offs = drop_off_state()
    st.markdown('<div class="step">Community drop-off map</div>', unsafe_allow_html=True)
    st.write(
        "This demo map is centered on Nairobi, Kenya. Community members can add useful "
        "drop-off areas so the map grows through local knowledge."
    )

    render_pin_map(drop_offs)

    st.markdown('<div class="map-list">', unsafe_allow_html=True)
    for point in drop_offs:
        accepts = ", ".join(point.get("accepts", []))
        st.markdown(
            f"""
            <div class="map-item">
                <strong>{point["name"]}</strong><br>
                <span class="muted">{accepts}</span><br>
                <span class="muted">{point["note"]}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_add_drop_off_area() -> None:
    st.markdown('<div class="step">Add a drop-off area</div>', unsafe_allow_html=True)
    st.write(
        "Click the exact location on the map. A red preview pin will appear there, then "
        "you can add its details for the community."
    )

    selected_location = st.session_state.get("selected_map_location")
    center = (
        [selected_location["lat"], selected_location["lon"]]
        if selected_location
        else [-1.2921, 36.8219]
    )
    community_map = folium.Map(
        location=center,
        zoom_start=14,
        tiles="CartoDB positron",
        control_scale=True,
    )
    LocateControl(
        auto_start=False,
        strings={"title": "Use my location"},
    ).add_to(community_map)

    for point in drop_off_state():
        folium.Marker(
            [point["lat"], point["lon"]],
            tooltip=point["name"],
            popup=folium.Popup(
                f"<strong>{point['name']}</strong><br>{point['note']}",
                max_width=280,
            ),
            icon=folium.Icon(color="green", icon="recycle", prefix="fa"),
        ).add_to(community_map)

    if selected_location:
        folium.Marker(
            [selected_location["lat"], selected_location["lon"]],
            tooltip="New drop-off point",
            icon=folium.Icon(color="red", icon="map-marker", prefix="fa"),
        ).add_to(community_map)

    map_event = st_folium(
        community_map,
        height=520,
        use_container_width=True,
        returned_objects=["last_clicked"],
        key="drop_off_picker",
    )
    clicked = map_event.get("last_clicked") if map_event else None
    if clicked:
        clicked_location = {
            "lat": float(clicked["lat"]),
            "lon": float(clicked["lng"]),
        }
        if clicked_location != selected_location:
            st.session_state.selected_map_location = clicked_location
            st.rerun()

    selected_location = st.session_state.get("selected_map_location")
    if selected_location:
        st.success(
            f"Location selected: {selected_location['lat']:.5f}, "
            f"{selected_location['lon']:.5f}"
        )
    else:
        st.info("Click anywhere on the map to place the new drop-off marker.")

    with st.form("add_drop_off_form", clear_on_submit=True):
        name = st.text_input("Place name", placeholder="Example: School gate bottle collection")
        category_labels = {
            "Recyclables": "Recycling Center",
            "Organic waste": "Organic Waste Bin",
            "Hazardous or hard-to-sort": "Recycling Center",
        }
        accepted_labels = st.multiselect(
            "What can people drop off here?",
            list(category_labels),
            default=["Recyclables"],
        )
        note = st.text_area(
            "Community note",
            placeholder="Opening times, landmark, contact person, or what should not be dropped here.",
        )
        submitted = st.form_submit_button("Add to community map", use_container_width=True)

    if submitted:
        if not selected_location:
            st.error("Please click the map to choose a location first.")
            return
        if not name.strip():
            st.error("Please add a place name.")
            return
        drop_off_state().append(
            {
                "name": name.strip(),
                "lat": selected_location["lat"],
                "lon": selected_location["lon"],
                "accepts": [category_labels[label] for label in accepted_labels] or ["Recycling Center"],
                "note": note.strip() or "Community-added drop-off area. Details still need verification.",
                "added_by": "Community",
            }
        )
        st.session_state.pop("selected_map_location", None)
        st.success("Added to the community map for this session.")

    render_community_map()


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
                    <span style="font-size:.75rem;font-weight:850;text-transform:uppercase;color:#eef8f2">Result</span>
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
    st.info("Open the Add Drop-Off Area tab to find or add a community disposal point.")


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
        render_live_points_progress("Points after this scan")
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
        render_live_points_progress("Points after this scan")
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


def render_home() -> None:
    render_hero()
    st.subheader("Our mission")
    st.write(
        "Sortify is designed as a learning app for cleaner communities: people scan waste, "
        "earn progress, and help map practical disposal points that are easy for neighbors to find."
    )
    st.markdown(
        """
        <div class="mission-grid">
            <div class="mission-tile">
                <strong>🔍 Learn by scanning</strong>
                <span class="muted">Every scan turns sorting guidance into a small action and a reward.</span>
            </div>
            <div class="mission-tile">
                <strong>🗺️ Build local knowledge</strong>
                <span class="muted">Community drop-off areas can be added by people who know the neighborhood.</span>
            </div>
            <div class="mission-tile">
                <strong>🌍 Support SDG 12</strong>
                <span class="muted">The goal is responsible consumption, cleaner streets, and better material recovery.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_community_map()


def render_journey() -> None:
    st.subheader("Journey, points, and rewards")
    render_dashboard()
    render_quests()
    render_guide()


def render_team_leaderboard() -> None:
    player = player_state()
    player_team = "Green Street Crew"
    teams = [
        {"name": "Mukuru Eco Stars", "points": 1280, "scans": 31},
        {"name": "Kibera Clean Team", "points": 1125, "scans": 27},
        {"name": "Mathare Recyclers", "points": 940, "scans": 23},
        {
            "name": player_team,
            "points": 720 + int(player.get("points", 0)),
            "scans": 18 + int(player.get("scans", 0)),
        },
        {"name": "Kayole Green Club", "points": 675, "scans": 16},
    ]
    teams.sort(key=lambda team: team["points"], reverse=True)
    player_rank = next(
        index for index, team in enumerate(teams, start=1) if team["name"] == player_team
    )
    player_score = next(team["points"] for team in teams if team["name"] == player_team)

    st.subheader("Team leaderboard")
    st.write(
        "Every scan adds to your team score. Work together, discover more waste categories, "
        "and climb the weekly community ranking."
    )
    st.markdown(
        f"""
        <div class="mission-grid">
            <div class="mission-tile"><strong>Your team</strong><span class="muted">{player_team}</span></div>
            <div class="mission-tile"><strong>Current rank</strong><span class="muted">#{player_rank} of {len(teams)} teams</span></div>
            <div class="mission-tile"><strong>Team score</strong><span class="muted">{player_score:,} weekly points</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    medals = {1: "1st", 2: "2nd", 3: "3rd"}
    for rank, team in enumerate(teams, start=1):
        player_class = " is-player" if team["name"] == player_team else ""
        team_note = "Your team" if team["name"] == player_team else "Community team"
        st.markdown(
            f"""
            <div class="leader-row{player_class}">
                <div class="leader-rank">{medals.get(rank, f"#{rank}")}</div>
                <div><strong>{team["name"]}</strong><br><span class="leader-label">{team_note}</span></div>
                <div><span class="leader-score">{team["points"]:,}</span><br><span class="leader-label">Points</span></div>
                <div class="leader-scans"><strong>{team["scans"]}</strong><br><span class="leader-label">Scans</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("Demo team rankings reset weekly. Your team score updates during this browser session.")


def render_profile() -> None:
    player = player_state()
    points = int(player.get("points", 0))
    badges = badges_for_player(player)
    badge_html = "".join(
        f'<span class="badge">{badge}</span>' for badge in badges
    ) or '<span class="badge">🚀 Ready to start</span>'
    st.subheader("Profile")
    st.markdown(
        f"""
        <div class="profile-header">
            <div class="avatar">♻️</div>
            <div>
                <h3 style="margin:.1rem 0">Community sorter</h3>
                <p class="muted" style="margin:.2rem 0">Level {level_for_points(points)} · {points} points · {player.get("scans", 0)} scans</p>
                <div class="bar"><span style="width:{progress_percent(points)}%"></span></div>
                <div class="badges">{badge_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="mission-grid">
            <div class="mission-tile"><strong>🔥 Streak</strong><span class="muted">{player.get("streak", 0)} active day(s)</span></div>
            <div class="mission-tile"><strong>🏷️ Categories</strong><span class="muted">{len(player.get("categories", []))} discovered</span></div>
            <div class="mission-tile"><strong>🗺️ Map points</strong><span class="muted">{len(drop_off_state())} community drop-off areas</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.header("System status")
    if os.path.exists(MODEL_PATH):
        st.success("AI model file found.")
    else:
        st.error("AI model file is missing.")
    if st.button("Reset game progress"):
        st.session_state.player = default_player()
        st.session_state.drop_offs = [dict(point) for point in DEFAULT_DROP_OFFS]
        st.session_state.pop("last_prediction", None)
        st.session_state.pop("last_demo_prediction", None)
        st.session_state.pop("selected_map_location", None)
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

home_tab, journey_tab, upload_tab, drop_tab, leaderboard_tab, profile_tab = st.tabs(
    ["Home", "Journey", "Upload Picture", "Add Drop-Off Area", "Team Leaderboard", "Profile"]
)

with home_tab:
    render_home()

with journey_tab:
    render_journey()

with upload_tab:
    if model is not None:
        render_scanner(model)
    elif TORCH_IMPORT_ERROR is not None:
        render_demo_scanner("Missing dependency: torch / torchvision.")
    else:
        render_demo_scanner("The model file is missing or could not be loaded.")

with drop_tab:
    render_add_drop_off_area()

with leaderboard_tab:
    render_team_leaderboard()

with profile_tab:
    render_profile()
