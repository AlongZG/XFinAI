import os


def wrap_path(path):
    os.makedirs(path, exist_ok=True)
    return path
