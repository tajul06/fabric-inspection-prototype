@echo off
setlocal

if not exist .venv (
    python -m venv .venv
)

call .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements_cuda.txt
pyinstaller --clean --noconfirm main.spec

echo CUDA build complete. Output in dist\
