"""Download Kubernetes OpenAPI specification from the official repository."""

from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent.parent
API_DIR = BASE_DIR / "data" / "k8s-api-specs"

SWAGGER_URL = (
    "https://raw.githubusercontent.com/kubernetes/kubernetes"
    "/master/api/openapi-spec/swagger.json"
)


def download_api_specs():
    API_DIR.mkdir(parents=True, exist_ok=True)
    dest = API_DIR / "swagger.json"

    print("Downloading Kubernetes OpenAPI spec (swagger.json)...")
    response = requests.get(SWAGGER_URL, timeout=120)
    response.raise_for_status()

    dest.write_text(response.text)
    size_mb = len(response.content) / 1024 / 1024
    print(f"Downloaded swagger.json ({size_mb:.1f} MB) to {API_DIR}")


if __name__ == "__main__":
    download_api_specs()
