# Sortify

Sortify is a Streamlit app for waste sorting and SDG 12 learning. It uses the
included EfficientNet model to classify a waste photo and turns each scan into a
small game loop.

## Features

- Camera or file upload waste scanner
- Four waste categories: hazardous, residual, organic, recyclable
- Home, journey, upload, community map, and profile tabs
- Disposal guidance and a demo community drop-off map centered on Nairobi, Kenya
- Points, levels, progress bar, streaks, and badges
- Points bar moves after every scan action
- Daily challenge to discover three waste categories
- Community form for adding local disposal or drop-off areas
- Quest log and sorting guide
- Animated icons, reward feedback, and demo scan mode
- Session-only progress with a reset button

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The model file `muell_scanner_model_efficientnet.pth` must stay in the project
folder so the scanner can load.

## Scanner troubleshooting

If the app says the real scanner is locked, the most common reason is that
PyTorch is not installed in the active Python environment. Run:

```powershell
pip install -r requirements.txt
```

The app still includes a demo scanner, so the game loop can be tested even before
the real model dependencies are available.
