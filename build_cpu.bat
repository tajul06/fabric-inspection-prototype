@echo off
setlocal

if not exist .venv (
    python -m venv .venv
)

call .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements_cpu.txt
pyinstaller --clean --noconfirm main.spec

echo CPU build complete. Output in dist\
