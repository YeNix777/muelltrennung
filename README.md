# muelltrennung
Repository für die Mülltrennungswebseite

Webseite zum lokal über Streamlit laufen lassen

Erste Anwendung:

sudo apt update
sudo apt install python3-full
(cd filepath)
python3 -m venv .venv
source .venv/bin/activate
pip install streamlit torch torchvision pillow
streamlit run app.py

Zukünftige Runs:

source .venv/bin/activate
streamlit run app.py
