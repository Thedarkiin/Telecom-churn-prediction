# utils.py

import os

def ensure_dirs(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)
        print(f" Created: {path}")

