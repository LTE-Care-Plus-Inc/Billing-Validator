# =========================================================
# HiRasmus Note Quality Checker
# - Fuzzy BT matching by NAME ONLY
# - Carries Email + Phone
# - Auto-assigns by UNIQUE BT (30/30/30/10)
# - FIXED length-mismatch bug
# =========================================================

import io
import re
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher

import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import streamlit as st


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="HiRasmus Note Quality Checker", layout="wide")
st.title("📝 HiRasmus Note Quality Checker")


# =========================================================
# PDF → TEXT
# =========================================================
def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(page.get_text("text") for page in doc)
    doc.close()
    return text


# =========================================================
# DATE NORMALIZATION
# =========================================================
def normalize_date(date_str: str) -> str:
    if not date_str:
        return ""
    for fmt in ("%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return date_str


# =========================================================
# TEXT CLEANING
# =========================================================
def clean_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00a0", " ")
    s = s.replace("\ufffd", "")
    return s.strip()


# =========================================================
# NAME NORMALIZATION
# =========================================================
def normalize_name(name: str) -> str:
    name = clean_text(name.lower())
    name = re.sub(r"[^a-z\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


# =========================================================
# FUZZY MATCH BT (NAME ONLY, RETURN META)
# =========================================================
def fuzzy_match_bt_with_meta(raw_bt: str, bt_ref: pd.DataFrame, threshold=0.8):
    if not raw_bt:
        return raw_bt, "", ""

    raw_norm = normalize_name(raw_bt)
    best_row = None
    best_score = 0.0

    for _, row in bt_ref.iterrows():
        score = SequenceMatcher(
            None,
            raw_norm,
            normalize_name(row["BT"])
        ).ratio()
        if score > best_score:
            best_score = score
            best_row = row

    if best_row is not None and best_score >= threshold:
        return (
            best_row["BT"],
            best_row.get("Email", ""),
            best_row.get("Phone", "")
        )

    return raw_bt, "", ""


# =========================================================
# PARSE NOTES
# =========================================================
def parse_notes_minimal(text: str):
    blocks = re.split(r"(?=Client\s*:)", text)
    rows = []

    for block in blocks:
        block = clean_text(block)
        if not block:
            continue

        def grab(pattern):
            m = re.search(pattern, block, re.I)
            return m.group(1).strip() if m else ""

        client = grab(r"Client\s*:\s*([^\n]+)")
        bt_raw = grab(r"Rendering Provider\s*:\s*([^\n]+)")
        raw_date = grab(
            r"Session Date\s*:\s*((?:\d{4}/\d{1,2}/\d{1,2})|(?:\d{1,2}/\d{1,2}/\d{4}))"
        )
        session_date = normalize_date(raw_date)

        if not client:
            continue

        data_rows = re.findall(
            r"\n\s*([A-Za-z][A-Za-z0-9\s,.\-’'()/]+?)\s+([0-9]{1,3})\s+([0-9]{1,3})\s*%?",
            block
        )
        has_session_data = bool(data_rows)

        summary_match = re.search(
            r"(Session Summary|Summary of Session)\s*:\s*(.+?)(?:\n[A-Z][a-zA-Z ]+?:|\Z)",
            block,
            re.I | re.S
        )
        has_session_summary = bool(summary_match and summary_match.group(2).strip())

        rows.append({
            "Session Date": session_date,
            "BT (Raw)": bt_raw,
            "Client": client,
            "Has Session Data": has_session_data,
            "Has Session Summary": has_session_summary,
            "Missing Session Data": not has_session_data,
            "Missing Session Summary": not has_session_summary,
        })

    return rows


# =========================================================
# ASSIGN BY UNIQUE BT
# =========================================================
def assign_by_unique_bt(df: pd.DataFrame, seed=42):
    bts = df["BT"].dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(bts)

    n = len(bts)
    n_max = int(n * 0.30)
    n_jerry = int(n * 0.30)
    n_dylan = int(n * 0.30)
    n_jesus = n - (n_max + n_jerry + n_dylan)

    assignments = (
        ["Max"] * n_max +
        ["Jerry"] * n_jerry +
        ["Dylan"] * n_dylan +
        ["Jesus"] * n_jesus
    )
    rng.shuffle(assignments)

    bt_map = dict(zip(bts, assignments))
    df["Assigned To (by BT)"] = df["BT"].map(bt_map).fillna("")
    return df


# =========================================================
# UI
# =========================================================
pdf_file = st.file_uploader("Upload HiRasmus Session Notes PDF", type=["pdf"])
bt_file = st.file_uploader("Upload BT Reference File (Excel or CSV)", type=["xlsx", "csv"])

if pdf_file and bt_file:
    with st.spinner("Processing…"):
        # Parse PDF
        text = pdf_bytes_to_text(pdf_file.read())
        df = pd.DataFrame(parse_notes_minimal(text)).reset_index(drop=True)
        df = df.copy().reset_index(drop=True)

        # Load BT reference
        bt_ref = (
            pd.read_csv(bt_file)
            if bt_file.name.endswith(".csv")
            else pd.read_excel(bt_file)
        )
        bt_ref = bt_ref.rename(columns=lambda x: x.strip())

        required_cols = {"BT", "Email", "Phone"}
        if not required_cols.issubset(bt_ref.columns):
            st.error("BT file must contain columns: BT, Email, Phone")
            st.stop()

        # -----------------------------
        # SAFE FUZZY MATCH (FIXED)
        # -----------------------------
        matched = [
            fuzzy_match_bt_with_meta(x, bt_ref)
            for x in df["BT (Raw)"].fillna("").tolist()
        ]

        matched_df = pd.DataFrame(
            matched,
            columns=["BT", "BT Email", "BT Phone"]
        )

        df = pd.concat([df, matched_df], axis=1)

        # Assign reviewers
        df = assign_by_unique_bt(df)

        # Split datasets
        mask = (
    df["Missing Session Data"].astype(bool).values |
    df["Missing Session Summary"].astype(bool).values
)

        issues_df = df.loc[mask].copy()
        good_df = df.loc[~mask].copy()

        def build_error(r):
            e = []
            if r["Missing Session Data"]:
                e.append("Missing Session Data")
            if r["Missing Session Summary"]:
                e.append("Missing Session Summary")
            return "; ".join(e)

        issues_df["Error"] = issues_df.apply(build_error, axis=1)

        for d in (issues_df, good_df, df):
            d.sort_values(["Session Date", "BT", "Client"], inplace=True, na_position="last")

    st.success("Processing complete.")
    st.dataframe(issues_df, use_container_width=True, height=600)

    # =========================================================
    # EXCEL EXPORT
    # =========================================================
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        issues_df[
            ["Session Date", "BT", "BT Email", "BT Phone", "Client", "Assigned To (by BT)", "Error"]
        ].to_excel(writer, index=False, sheet_name="Note Issues")

        good_df[
            ["Session Date", "BT", "BT Email", "BT Phone", "Client", "Assigned To (by BT)"]
        ].to_excel(writer, index=False, sheet_name="Good Notes")

        df.to_excel(writer, index=False, sheet_name="All Notes")

    output.seek(0)

    st.download_button(
        "⬇️ Download Note Audit Report",
        data=output,
        file_name="hirasmus_note_audit_fuzzy_bt.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
