"""
LeafScan AI — Model Downloader
Run this once before starting the app:
    python download_model.py
"""

import os
import sys

try:
    import gdown
except ImportError:
    print("Installing gdown...")
    os.system(f"{sys.executable} -m pip install gdown")
    import gdown

# ── Model file IDs from Google Drive ──────────────────────────────────────────
MODELS = {
    "plant_disease_model.keras": "1fEnq8VQQRR-RiD4dPq2cFtwYOZH76JX9",
    "plant_disease_model.tflite": "1dDnzzDQqiKgY0Mv0W8AO5FSnfxQxZsQg",
}

def download(filename, file_id):
    if os.path.exists(filename):
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"✅ {filename} already exists ({size_mb:.1f} MB) — skipping.")
        return
    print(f"⬇️  Downloading {filename} from Google Drive...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, filename, quiet=False)
    if os.path.exists(filename):
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"✅ {filename} downloaded successfully ({size_mb:.1f} MB)")
    else:
        print(f"❌ Failed to download {filename}. Check the file ID or sharing permissions.")

if __name__ == "__main__":
    print("=" * 50)
    print("  LeafScan AI — Model Downloader")
    print("=" * 50)

    # Download only keras by default, tflite is optional
    print("\n[1/2] Keras model (full, recommended):")
    download("plant_disease_model.keras", MODELS["plant_disease_model.keras"])

    print("\n[2/2] TFLite model (lite/mobile, optional):")
    ans = input("Download TFLite model too? (y/n): ").strip().lower()
    if ans == "y":
        download("plant_disease_model.tflite", MODELS["plant_disease_model.tflite"])
    else:
        print("⏭  Skipped TFLite model.")

    print("\n" + "=" * 50)
    print("  All done! Now run:  streamlit run app.py")
    print("=" * 50)
