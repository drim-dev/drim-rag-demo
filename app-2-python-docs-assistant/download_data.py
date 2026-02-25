"""Download Python documentation RST files from the CPython repository."""

import io
import os
import zipfile

import requests

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "python-docs")

CPYTHON_ZIP_URL = "https://github.com/python/cpython/archive/refs/heads/main.zip"

# Most important stdlib modules for a useful RAG corpus
LIBRARY_MODULES = [
    "os", "sys", "pathlib", "shutil",
    "asyncio", "concurrent.futures", "threading", "multiprocessing",
    "collections", "itertools", "functools", "operator",
    "json", "csv", "sqlite3", "pickle",
    "re", "string", "textwrap",
    "typing", "dataclasses", "abc", "enum",
    "unittest", "doctest",
    "logging", "warnings",
    "argparse", "configparser",
    "subprocess", "os.path",
    "datetime", "time", "calendar",
    "http.client", "http.server", "urllib.request", "urllib.parse",
    "socket", "ssl",
    "io", "contextlib",
    "pdb", "traceback",
    "importlib", "pkgutil",
    "copy", "pprint",
    "hashlib", "secrets",
    "struct", "array",
]

LIBRARY_FILES = {f"library/{mod.replace('.', '')}.rst" for mod in LIBRARY_MODULES}
# os.path is documented inside os module, handle separately
LIBRARY_FILES.discard("library/os.path.rst")


def _should_keep(path_in_doc: str) -> bool:
    """Decide whether to keep an RST file from the Doc/ directory."""
    if path_in_doc.startswith("tutorial/"):
        return path_in_doc.endswith(".rst")
    if path_in_doc in LIBRARY_FILES:
        return True
    return False


def download_docs() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading CPython source archive (this may take a minute)...")
    resp = requests.get(CPYTHON_ZIP_URL, timeout=120, stream=True)
    resp.raise_for_status()

    content = io.BytesIO(resp.content)
    print(f"Downloaded {len(resp.content) / 1024 / 1024:.1f} MB")

    saved = 0
    total_bytes = 0

    with zipfile.ZipFile(content) as zf:
        # The archive root is cpython-main/
        doc_prefix = None
        for name in zf.namelist():
            if "/Doc/" in name and doc_prefix is None:
                doc_prefix = name[: name.index("/Doc/") + len("/Doc/")]
                break

        if not doc_prefix:
            print("ERROR: Could not find Doc/ directory in archive")
            return

        for entry in zf.namelist():
            if not entry.startswith(doc_prefix) or not entry.endswith(".rst"):
                continue

            relative = entry[len(doc_prefix):]
            if not _should_keep(relative):
                continue

            raw = zf.read(entry)
            out_path = os.path.join(DATA_DIR, relative)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            with open(out_path, "wb") as f:
                f.write(raw)

            saved += 1
            total_bytes += len(raw)

    print(f"\nDone: {saved} RST files saved")
    print(f"Total size: {total_bytes / 1024:.1f} KB")
    print(f"Location: {DATA_DIR}")


if __name__ == "__main__":
    print("Downloading Python documentation...\n")
    download_docs()
