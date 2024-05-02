# !/bin/bash

python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
echo 'OPENAI_SECRET_KEY=""' > local_settings.py
