import subprocess
import webbrowser
import time
import os

# Correct Linux path
venv_python = os.path.join("venv", "bin", "python")

# Start backend
backend = subprocess.Popen([venv_python, "backend.py"])

# Start sniffer
sniffer = subprocess.Popen([venv_python, "sniffer.py"])

# Wait for backend
time.sleep(3)

# Open dashboard
webbrowser.open("http://127.0.0.1:8000")

print("NIDS is running...")

try:
    backend.wait()
    sniffer.wait()
except KeyboardInterrupt:
    print("Shutting down...")
    backend.terminate()
    sniffer.terminate()
