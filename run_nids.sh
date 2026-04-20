#!/bin/bash

# 1. Activate virtual environment
source venv/bin/activate

# 2. Start packet capture (background)
python live_ids.py &

# 3. Start NIDS backend
python main.py
