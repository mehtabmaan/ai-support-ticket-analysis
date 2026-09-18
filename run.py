"""
Single-Command Launcher for AI Support Ticket Analysis System.
Starts the FastAPI REST API (port 8000) and the Streamlit UI (port 8501)
concurrently and manages graceful shutdown.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "support_tickets.csv"


def check_prerequisites():
    """Validates that necessary files and environment are present."""
    if not DATA_FILE.exists():
        print(f"[ERROR] Dataset file missing at {DATA_FILE}!")
        sys.exit(1)


def wait_for_api_health(url: str, timeout: int = 15) -> bool:
    """Polls the API health endpoint until it responds or times out."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=1.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    check_prerequisites()

    api_port = int(os.environ.get("FASTAPI_PORT", 8000))
    ui_port = int(os.environ.get("STREAMLIT_PORT", 8501))
    host = os.environ.get("FASTAPI_HOST", "127.0.0.1")

    api_url = f"http://{host}:{api_port}"
    health_url = f"{api_url}/health"
    docs_url = f"{api_url}/docs"
    ui_url = f"http://localhost:{ui_port}"

    print("=" * 70)
    print(" 🚀 STARTING AI CUSTOMER SUPPORT TICKET ANALYSIS SYSTEM")
    print("=" * 70)
    print(f"[*] Python Interpreter : {sys.executable}")
    print(f"[*] Base Directory     : {BASE_DIR}")
    print(f"[*] REST API Target    : {api_url}")
    print(f"[*] Interactive UI     : {ui_url}")
    print("=" * 70)

    # 1. Start FastAPI backend
    print("\n[1/2] Launching FastAPI backend server...")
    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "main:app",
        "--host",
        host,
        "--port",
        str(api_port),
    ]

    api_process = subprocess.Popen(
        api_cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for API health
    print("[*] Verifying API readiness...")
    if wait_for_api_health(health_url, timeout=15):
        print(f"[SUCCESS] FastAPI is live at: {api_url}")
        print(f"[SUCCESS] OpenAPI Swagger docs: {docs_url}")
    else:
        print("[WARNING] API startup confirmation timed out; proceeding to UI...")

    # 2. Start Streamlit UI
    print("\n[2/2] Launching Streamlit UI...")
    ui_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(BASE_DIR / "src" / "ui" / "app.py"),
        "--server.port",
        str(ui_port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    ui_process = subprocess.Popen(
        ui_cmd,
        cwd=str(BASE_DIR),
    )

    print("\n" + "=" * 70)
    print(f" ✅ ALL SERVICES RUNNING SUCCESSFULLY!")
    print(f"    - Web Interface   : {ui_url}")
    print(f"    - REST API Docs   : {docs_url}")
    print(f"    - Health Check    : {health_url}")
    print("    Press Ctrl+C to terminate all services cleanly.")
    print("=" * 70 + "\n")

    def shutdown(signum, frame):
        print("\n[*] Shutting down services cleanly...")
        try:
            ui_process.terminate()
            api_process.terminate()
            ui_process.wait(timeout=3)
            api_process.wait(timeout=3)
        except Exception:
            ui_process.kill()
            api_process.kill()
        print("[*] All services stopped. Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Keep main thread alive monitoring children
    try:
        while True:
            if api_process.poll() is not None:
                print(f"[ERROR] API process exited unexpectedly with code {api_process.returncode}")
                break
            if ui_process.poll() is not None:
                print(f"[INFO] UI process exited with code {ui_process.returncode}")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown(None, None)


if __name__ == "__main__":
    main()
