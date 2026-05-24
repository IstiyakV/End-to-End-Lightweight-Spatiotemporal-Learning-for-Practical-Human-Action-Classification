import json
from pathlib import Path

# Always save to NewCode/gui_settings.json regardless of where terminal is opened
SETTINGS_FILE = Path(__file__).resolve().parent.parent / "gui_settings.json"

def load_settings():
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_setting(key, value):
    settings = load_settings()
    settings[key] = str(value)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)
