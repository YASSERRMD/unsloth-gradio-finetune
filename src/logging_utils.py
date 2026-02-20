import os


def tail_file(path: str, n: int = 200) -> str:
    """Return last n lines of a log file."""
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        lines = f.readlines()
    return "".join(lines[-n:])
