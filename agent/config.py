import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".astroagent"
CONFIG_FILE = CONFIG_DIR / "config.json"


def ensure_config_dir():
    CONFIG_DIR.mkdir(exist_ok=True)


def load_config() -> dict:
    ensure_config_dir()
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())
    return {}


def save_config(config: dict):
    ensure_config_dir()
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def get_api_key() -> str | None:
    return load_config().get("openai_api_key")


def set_api_key(key: str):
    config = load_config()
    config["openai_api_key"] = key
    save_config(config)


def clear_config():
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
