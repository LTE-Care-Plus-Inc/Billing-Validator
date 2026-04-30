# Billing-Validator (Streamlit)

This project can be run as a Streamlit app inside Docker using a hardened non-root container.

## Run locally (without Docker)

1) Create and activate a virtual environment

Windows (PowerShell):

`python -m venv .venv`

`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

`.\.venv\Scripts\Activate.ps1`

If script activation is blocked by company policy, use one of these alternatives:

- Run tools directly without activation:
  `.\.venv\Scripts\python -m pip install --upgrade pip`
  `.\.venv\Scripts\python -m streamlit run billing_checker.py`
- Or use Command Prompt activation:
  `.\.venv\Scripts\activate.bat`

2) Install dependencies

`pip install --upgrade pip`

`pip install pandas numpy streamlit openpyxl XlsxWriter PyMuPDF`

3) Start the app

`streamlit run billing_checker.py`

4) Open in browser

`http://localhost:8501`

## Run with Docker

1) Build image

`docker build -t billing-validator-streamlit .`

2) Run container

`docker run --rm -p 8501:8501 billing-validator-streamlit`

3) Open in browser

`http://localhost:8501`

