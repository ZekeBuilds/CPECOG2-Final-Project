"""Download UJIIndoorLoc from the UCI repository into data/raw/UJIndoorLoc/.

Run from the project root:  python -m scripts.download_data
Skips the download if the CSV files are already there.
"""
import io
import urllib.request
import zipfile

from src.data import RAW_DIR

URL = "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip"
FILES = ("trainingData.csv", "validationData.csv")


def main():
    if all((RAW_DIR / f).exists() for f in FILES):
        print(f"Already downloaded: {RAW_DIR}")
        return
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=120) as resp:
        payload = resp.read()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for f in FILES:
            # Write each CSV by name only, so nothing in the archive can choose its own path.
            (RAW_DIR / f).write_bytes(z.read(f"UJIndoorLoc/{f}"))
    print(f"Saved {', '.join(FILES)} to {RAW_DIR}")


if __name__ == "__main__":
    main()
