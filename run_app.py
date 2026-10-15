import os
import sys
import socket
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "webapp" / "backend"))

import uvicorn
from server import app

def find_available_port(preferred_port=8050):
    for port in [preferred_port, 8080, 8501, 8888, 5000]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

PORT = find_available_port(8050)

if __name__ == "__main__":
    print(f"\n=======================================================")
    print(f"  SCANIA TRUCKS APS PREDICTIVE MAINTENANCE SYSTEM")
    print(f"  Live UI Dashboard URL: http://127.0.0.1:{PORT}")
    print(f"  API Documentation:    http://127.0.0.1:{PORT}/docs")
    print(f"=======================================================\n")
    
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
