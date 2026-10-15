import os
import sys
import socket
import threading
import time
from pathlib import Path

import httpx
import pandas as pd
import uvicorn

# Ensure imports and relative file paths (like scania_aps_pipeline.joblib and CSVs)
# resolve correctly when running from the parent VS Code workspace folder.
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app import app  # noqa: E402


def load_test_features() -> pd.DataFrame:
    """Load the test dataset and drop the target column ('class') if present."""
    processed_csv = BASE_DIR / "aps_failure_test_set_processed_8bit.csv"
    raw_csv = BASE_DIR / "aps_failure_test_set.csv"

    if processed_csv.exists():
        df = pd.read_csv(processed_csv, na_values=["na", "NA", "NaN"])
    elif raw_csv.exists():
        try:
            df = pd.read_csv(raw_csv, na_values=["na", "NA", "NaN"])
            if df.shape[1] == 1:
                df = pd.read_csv(raw_csv, skiprows=20, na_values=["na", "NA", "NaN"])
        except Exception:
            df = pd.read_csv(raw_csv, skiprows=20, na_values=["na", "NA", "NaN"])
    else:
        raise FileNotFoundError(
            f"Could not find test dataset CSV in {BASE_DIR}"
        )

    X_test = df.drop(columns=["class"], errors="ignore")
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X_test


def get_free_port() -> int:
    """Bind to port 0 on localhost to obtain an available ephemeral port."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


PORT = get_free_port()


def run_server() -> None:
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    X_test = load_test_features()

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2)

    sample_features = {k: float(v) for k, v in X_test.iloc[0].to_dict().items()}
    sample_payload = {"features": sample_features}

    live_response = httpx.post(
        f"http://127.0.0.1:{PORT}/predict",
        json=sample_payload,
        timeout=30.0,
    )

    print(f"Live Server URL: http://127.0.0.1:{PORT}/docs")
    try:
        print("Live Prediction Response:", live_response.json())
    except Exception:
        print("Live Prediction Status:", live_response.status_code, live_response.text)

    print("\nServer is running. Press Ctrl+C to stop.")
    try:
        while server_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")