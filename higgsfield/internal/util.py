import re
from pathlib import Path

from fabric.config import os


regex = re.compile("^[a-zA-Z_][a-zA-Z0-9_]*$")


def check_name(name: str):
    if len(name) < 1 or len(name) > 20:
        raise ValueError("Name must be between 1 and 20 characters long")

    if not regex.match(name):
        raise ValueError("Name must match regex ^[a-zA-Z_][a-zA-Z0-9_]*$")

    return name


def wd_path() -> Path:
    return Path.cwd()


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def templates_path() -> Path:
    return Path(ROOT_DIR) / "static" / "templates"
