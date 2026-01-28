# ===========================
# app.py (UPDATED)
# Matching unchanged; time pipeline centralized
# ===========================
import io
import re
import unicodedata
from typing import Any
from difflib import SequenceMatcher

from utils import (
    normalize_date,
    normalize_session_time, 
    parse_session_time_range,
    normalize_time_range,
)

import fitz
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="HiRasmus Billing Session Checker",
    layout="wide",
)
st.title("HiRasmus Billing Session Checker")

# -----------------------------
# GLOBAL CONFIG
# -----------------------------
STATUS_REQUIRED = "Transferred to AlohaABA"
SESSION_REQUIRED = "1:1 BT Direct Service"

MIN_MINUTES = 53
MAX_MINUTES = 360
BILLING_TOL_DEFAULT = 8
DAILY_MAX_MINUTES = 480

TIME_ADJ_COL = "Adult Caregiver’s Signature Approval for Time Adjustment signature"

REQ_COLS = [
    "Status",
    "AlohaABA Appointment ID",
    "Client",
    "Duration",
    "Session",
    "Adult Caregiver signature time",
    "User",
    "Start date",
]

USE_TIME_ADJ_OVERRIDE = True

DATE_RE = r"(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4})"

# =========================================================
# Excel-safe sanitizer
# =========================================================
_ILLEGAL_XL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def excel_safe_str(v: Any) -> Any:
    if v is None:
        return v
    if isinstance(v, float) and pd.isna(v):
        return v
    s = str(v)
    s = _ILLEGAL_XL_CHARS_RE.sub("", s)
    return s


def excel_sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        if df2[c].dtype == object:
            df2[c] = df2[c].map(excel_safe_str)
    return df2


# -----------------------------
# PDF → TEXT (TOOLS TAB)
# -----------------------------
def pdf_bytes_to_text(pdf_bytes: bytes, preserve_layout: bool = True) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    all_text = []

    for page_num in range(total_pages):
        page = doc[page_num]
        if preserve_layout:
            text = page.get_text(
                "text",
                flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES,
            )
        else:
            text = page.get_text("text")

        all_text.append(f"--- Page {page_num + 1} ---\n{text}\n\n")

    doc.close()
    return "".join(all_text)


# =========================================================
# Provider signature parsing + validation
# =========================================================
def _clean_pdf_line(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\u00a0", " ").replace("􀀀", " ")
    s = "".join(ch for ch in s if ch.isprintable())
    s = " ".join(s.split())
    return s.strip()


def extract_provider_signature_candidates(block_text: str) -> list[str]:
    label_re = re.compile(
        r"Provider\s+Signatures?\s*(?:/\s*)?Credentials\s+and\s+Date\s*:?\s*",
        re.I,
    )

    cands: list[str] = []
    for m in label_re.finditer(block_text):
        tail = block_text[m.end(): m.end() + 800]
        if not tail:
            continue

        lines = tail.splitlines()
        first_line = _clean_pdf_line(lines[0]) if len(lines) >= 1 else ""

        rest_lines = [_clean_pdf_line(x) for x in lines[1:10]]
        rest_lines = [x for x in rest_lines if x]

        pieces = []
        if first_line:
            pieces.append(first_line)
        if rest_lines:
            pieces.append(rest_lines[0])

        cand = _clean_pdf_line(" ".join(pieces))
        if cand:
            cands.append(cand)

    return cands


def any_signature_line_valid(block_text: str) -> tuple[bool, list[str]]:
    cands = extract_provider_signature_candidates(block_text)

    for cand in cands:
        if not re.search(r"R?BT", cand, re.I):
            continue
        if not re.search(r"\b\d{1,4}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{1,4}\b", cand):
            continue
        if not any(ch.isalpha() for ch in cand):
            continue
        return True, cands

    return False, cands


# =========================================================
# NOTE PARSER
# =========================================================
def parse_notes(text: str):
    blocks = re.split(r"(?=Client\s*:)", text)
    results = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        client_match = re.search(r"Client\s*:\s*([^\n]+)", block)
        client_name = client_match.group(1).strip() if client_match else ""
        if not client_name:
            continue

        provider_match = re.search(r"Rendering Provider\s*:\s*([^\n]+)", block)
        provider = provider_match.group(1).strip() if provider_match else ""

        dob_match = re.search(rf"Date\s*of\s*Birth\s*[:\-]?\s*({DATE_RE})", block, re.I)
        dob = normalize_date(dob_match.group(1)) if dob_match else ""

        gender_match = re.search(r"Gender\s*:\s*([^\n\r]+)", block, re.I)
        gender_raw = gender_match.group(1).strip() if gender_match else ""

        g = gender_raw.strip().lower()
        if g in ("male", "m", "man", "boy", "男性", "男"):
            gender = "Male"
        elif g in ("female", "f", "woman", "girl", "女性", "女"):
            gender = "Female"
        else:
            gender = gender_raw

        diagnosis = ""
        dx_match = re.search(
            r"Diagnosis Code\s*\(\s*(?:ICD|IDC)\s*[-\s]*10\s*\)\s*:\s*([A-Za-z0-9\.\s]+)",
            block,
            re.I,
        )
        if dx_match:
            raw = dx_match.group(1)
            icd_match = re.search(r"[A-Za-z]\d{2}(?:\.\d+)?", raw, re.I)
            if icd_match:
                diagnosis = icd_match.group(0).upper()

        insurance_match = re.search(r"Primary Insurance\s*:\s*([^\n]+)", block)
        primary_insurance = insurance_match.group(1).strip() if insurance_match else ""

        ins_id_match = re.search(r"Insurance ID\s*:\s*([A-Z0-9]+)", block, re.I)
        insurance_id = ins_id_match.group(1).strip() if ins_id_match else ""

        date_match = re.search(rf"Session Date\s*:\s*{DATE_RE}", block, re.I)
        session_date = normalize_date(date_match.group(1)) if date_match else ""

        # SAME-LINE ONLY policy preserved (unchanged)
        raw_session_time = ""
        m_time = re.search(r"(?im)^\s*Session Time\s*:\s*(.*)\s*$", block)
        if m_time:
            raw_session_time = m_time.group(1).strip()
            if raw_session_time in ("-", "–", "—"):
                raw_session_time = ""


        session_time = normalize_time_range(raw_session_time)
        location_match = re.search(r"Session Location\s*:\s*([^\n]+)", block)
        session_location = location_match.group(1).strip() if location_match else ""

        present_text = ""
        pos = block.lower().find("present at session")
        if pos != -1:
            present_text = block[pos: pos + 400]

        present_client = bool(re.search(r"\bClient\b", present_text, re.I))
        present_bt = bool(re.search(r"\b(BT/RBT|RBT/BT)\b", present_text, re.I))
        present_caregiver = bool(re.search(r"\bAdult Caregiver\b", present_text, re.I))
        present_sibling = bool(re.search(r"\bSibling(s)?\b", present_text, re.I))

        maladaptive_section = ""
        section_match = re.search(
            r"Maladaptive Status\s*:\s*(.*?)(?:\n[A-Z][a-zA-Z ]+?:|\Z)",
            block,
            re.S,
        )
        if section_match:
            maladaptive_section = section_match.group(1).strip()

        maladaptive_behaviors = []
        if maladaptive_section:
            for line in maladaptive_section.splitlines():
                clean = line.strip()
                if not clean:
                    continue

                lower = clean.lower()
                if (
                    lower.endswith(":")
                    or "continues to display" in lower
                    or "in the following areas" in lower
                    or "maladaptive status" in lower
                    or "other maladaptive behaviors" in lower
                ):
                    continue

                clean = re.sub(r"[•▪◦\-–—\uf0b7\uf0a7]+", "", clean).strip()

                if len(clean.split()) > 6 and not clean.lower().startswith("other"):
                    continue

                maladaptive_behaviors.append(clean.lower())

        other_selected = any(b == "other" or b.startswith("other ") for b in maladaptive_behaviors)
        other_desc_match = re.search(r"Other maladaptive behaviors\s*:\s*(.+)", block, re.I)
        other_maladaptive_present = bool(other_desc_match and other_desc_match.group(1).strip())

        data_rows = re.findall(
            r"\n\s*([A-Za-z][A-Za-z0-9\s,.\-’'()/]+?)\s+([0-9]{1,3})\s+([0-9]{1,3})\s*%?\s*(?=\n|$)",
            block
        )
        data_collected = len(data_rows) > 0

        summary_match = re.search(
            r"(Session Summary|Summary of Session)\s*:\s*(.+?)(?:\n[A-Z][a-zA-Z ]+?:|\Z)",
            block,
            re.I | re.S,
        )
        session_summary_present = bool(summary_match and summary_match.group(2).strip())

        bt_attestation_present = bool(
            re.search(
                r"\battest\s+that\s+the\s+session\s+summary\s+is\s+accurate\s+and\s+correct\b",
                block,
                re.I,
            )
        )

        revision_attestation_present = bool(
            re.search(
                r"I\s+attest\s+the\s+revision/edit\s+made\s+to\s+this\s+note\s+as\s+signed\s+below\s+is\s+accurate\s+and\s+true",
                block,
                re.I,
            )
        )

        outcome_yes = bool(re.search(r"Outcome of Treatment.*?:\s*Yes", block, re.I | re.S))

        sig_valid, sig_cands = any_signature_line_valid(block)
        provider_signature_present = len(sig_cands) > 0
        provider_signature_valid = sig_valid

        compliance_errors = []

        if not dob:
            compliance_errors.append("Missing DOB")
        if not gender:
            compliance_errors.append("Missing Gender")
        if not diagnosis:
            compliance_errors.append("Missing ICD-10")
        if diagnosis and len(diagnosis) < 3:
            compliance_errors.append("Invalid ICD-10 code (truncated)")
        if not primary_insurance:
            compliance_errors.append("Missing Primary Insurance")
        if not insurance_id:
            compliance_errors.append("Missing Insurance ID")

        # NOTE: time must be normalized successfully; otherwise it is "missing" for compliance
        if not session_time:
            compliance_errors.append("Missing Session Time")

        if not session_location:
            compliance_errors.append("Missing Session Location")

        if not maladaptive_behaviors:
            compliance_errors.append("No maladaptive behaviors listed")

        if other_selected and not other_maladaptive_present:
            compliance_errors.append("Other maladaptive behavior selected but no description provided")

        if not data_collected:
            compliance_errors.append("Missing Session Data")
        if not session_summary_present:
            compliance_errors.append("Missing session summary narrative")
        if not outcome_yes:
            compliance_errors.append("Outcome of Treatment not Yes")

        if not bt_attestation_present:
            compliance_errors.append("Missing BT/RBT attestation statement")

        if not provider_signature_present:
            compliance_errors.append("Missing provider signature section")
        elif not provider_signature_valid:
            compliance_errors.append("Provider signature present but invalid format (must include Name, BT/RBT, and date)")

        if not present_client:
            compliance_errors.append("Attendance: Client not present")
        if not present_bt:
            compliance_errors.append("Attendance: BT/RBT not present")
        if not (present_caregiver or present_sibling):
            compliance_errors.append("Attendance: Parent/Caregiver or Sibling not present")

        results.append(
            {
                "Client": client_name,
                "Rendering Provider": provider,
                "Session Date": session_date,
                "Date of Birth": dob,
                "Gender": gender,
                "Diagnosis Code": diagnosis,
                "Primary Insurance": primary_insurance,
                "Insurance ID": insurance_id,
                "Session Time": session_time,
                "Session Location": session_location,
                "Present_Client": present_client,
                "Present_Adult_Caregiver": present_caregiver,
                "Present_Sibling": present_sibling,
                "Present_BT_RBT": present_bt,
                "Maladaptive Behaviors": maladaptive_behaviors,
                "Other Selected": other_selected,
                "Other Maladaptive Provided": other_maladaptive_present,
                "Outcome Yes": outcome_yes,
                "Data Collected": data_collected,
                "Session Summary Present": session_summary_present,
                "BT Attestation Present": bt_attestation_present,
                "Provider Signature Present": provider_signature_present,
                "Provider Signature Valid": provider_signature_valid,
                "Provider Signature Candidates": sig_cands,
                "Revision Attestation Present": revision_attestation_present,
                "Compliance Errors": compliance_errors,
                "PASS": len(compliance_errors) == 0,
            }
        )

    return results


def notes_to_excel_bytes(results, sheet_name="Notes") -> bytes:
    df = pd.DataFrame(results)
    df = excel_sanitize_df(df)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)
    return output.getvalue()


# -----------------------------
# SHARED HELPERS
# -----------------------------
def read_any(file):
    if file is None:
        return None
    if file.name.lower().endswith((".csv", ".txt")):
        return pd.read_csv(file, dtype=str)
    return pd.read_excel(file, dtype=str, engine="openpyxl")


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .replace({"nan": np.nan, "": np.nan})
            )
    return df


def ensure_cols(df: pd.DataFrame, cols, label: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"**{label}** is missing required columns: {missing}")
        return False
    return True


def parse_duration_to_minutes(d: Any) -> float:
    if pd.isna(d):
        return np.nan
    s = str(d).strip()
    if not s or ":" not in s:
        return np.nan
    try:
        parts = [float(x) for x in s.split(":")]
        if len(parts) == 3:
            h, m, sec = parts
            return h * 60 + m + sec / 60
        elif len(parts) == 2:
            h, m = parts
            return h * 60 + m
    except Exception:
        return np.nan
    return np.nan


def normalize_client_name_for_match(name: Any) -> str:
    return normalize_name(name)


def export_excel(df: pd.DataFrame) -> bytes:
    df2 = excel_sanitize_df(df)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df2.to_excel(w, index=False, sheet_name="All")
    return buf.getvalue()


def within_time_tol(sig_ts, base_ts, tol_early_min):
    if pd.isna(sig_ts) or pd.isna(base_ts):
        return False
    diff_min = (sig_ts - base_ts).total_seconds() / 60.0
    return diff_min > tol_early_min


def normalize_name(name: Any) -> str:
    if pd.isna(name):
        return ""
    s = str(name).strip()
    if not s:
        return ""
    parts = s.replace(",", " ").split()
    parts = [p.capitalize() for p in parts if p]
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    last = parts[-1]
    return f"{last}, {first}"


# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("Settings")

with st.sidebar.expander("Signature Settings", expanded=False):
    SIG_TOL_EARLY = st.number_input(
        "Signature early tolerance (minutes, negative)",
        value=-8,
        step=1,
        help="Earliest allowed time before session end (e.g., -15 means 15 minutes before).",
    )

with st.sidebar.expander("Duration / Billing Settings", expanded=False):
    BILLING_TOL = st.number_input(
        "Billing duration tolerance (minutes)",
        value=BILLING_TOL_DEFAULT,
        min_value=0,
        step=1,
        help=(
            "If a session's duration is outside the base range "
            f"({MIN_MINUTES}-{MAX_MINUTES} min) by at most this many minutes, "
            "it will still be treated as OK."
        ),
    )
    DAILY_TOL = st.number_input(
        "Daily 8-hour limit tolerance (minutes)",
        value=8,
        min_value=0,
        step=1,
        help="How many extra minutes a BT can work over 8 hours in one day before it is flagged."
    )

st.sidebar.header("BT Contacts (optional)")
bt_contacts_file = st.sidebar.file_uploader(
    "Upload BT Contacts (BT Name, Phone, Email)",
    type=["csv", "xlsx"],
    key="bt_contacts",
)

with st.expander("Instructions", expanded=False):
    st.markdown(
        """
This tool reviews **BT 1:1 Direct Service sessions from HiRasmus** and checks whether each session meets
core billing-compliance requirements.

There are **three tabs** in this app:

- **Tools – Extract External Sessions**: Upload the *Session Notes PDF* from HiRasmus.
  The app parses it and builds an **External Session List** and stores it for use in the Session Checker.

- **Session Checker**: Upload the HiRasmus sessions Excel export and run all compliance checks.

- **Billed Checker**: Upload Aloha billing status and see billed/unbilled/no-match.
        """
    )

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "1️⃣ Tools – Extract External Sessions",
    "2️⃣ Session Checker",
    "3️⃣ Billed Checker",
])

if "external_sessions_df" not in st.session_state:
    st.session_state["external_sessions_df"] = None

# =========================================================
# TAB 1
# =========================================================
with tab1:
    st.header("Session Notes PDF → External Session List")

    pdf_file = st.file_uploader(
        "Upload HiRasmus Session Notes PDF",
        type=["pdf"],
        key="notes_pdf",
    )

    if pdf_file is not None:
        try:
            text = pdf_bytes_to_text(pdf_file.read(), preserve_layout=True)
            results = parse_notes(text)

            if not results:
                st.warning("No notes found in the PDF. Make sure it contains 'Client:' blocks.")
            else:
                notes_df = pd.DataFrame(results)
                st.success(f"Parsed {len(notes_df)} notes from PDF.")

                with st.expander("Preview parsed notes (compliance)", expanded=False):
                    preview_cols = [
                        "Client",
                        "Session Date",
                        "Session Time",
                        "Session Location",
                        "PASS",
                        "Compliance Errors",
                    ]
                    preview_cols = [c for c in preview_cols if c in notes_df.columns]
                    st.dataframe(notes_df[preview_cols].head(120), use_container_width=True)

                cols = [
                    "Client",
                    "Session Time",
                    "Session Date",
                    "Present_Client",
                    "Present_Adult_Caregiver",
                    "Present_Sibling",
                    "Present_BT_RBT",
                    "PASS",
                    "Compliance Errors",
                ]
                existing_cols = [c for c in cols if c in notes_df.columns]

                ext_df = notes_df[existing_cols].copy()

                ext_df = ext_df.rename(
                    columns={
                        "Present_Adult_Caregiver": "Present_ParentCaregiver",
                        "PASS": "Note Parse PASS",
                        "Compliance Errors": "Note Compliance Errors",
                    }
                )

                ext_df = normalize_cols(ext_df)

                ext_df = ext_df[ext_df["Client"].notna()].copy()

                st.session_state["external_sessions_df"] = ext_df

                st.info(
                    f"Generated {len(ext_df)} external session rows. "
                    "These will be used automatically in the **Session Checker** tab."
                )

                ext_xlsx = notes_to_excel_bytes(ext_df.to_dict(orient="records"), sheet_name="External Sessions")
                st.download_button(
                    "⬇️ Download External Session List (.xlsx)",
                    data=ext_xlsx,
                    file_name="external_sessions_from_pdf.xlsx",
                )

        except Exception as e:
            st.error(f"Error during PDF processing: {e}")


# =========================================================
# TAB 2: SESSION CHECKER
# =========================================================
def has_time_adjust_sig(row) -> bool:
    if TIME_ADJ_COL not in row.index:
        return False
    val = row.get(TIME_ADJ_COL)
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    return s not in ("", "nan")


def duration_ok_base(row) -> bool:
    m = row.get("Actual Minutes")
    if pd.isna(m):
        return False
    if m < MIN_MINUTES:
        return False
    if m <= MAX_MINUTES:
        return True
    return (m - MAX_MINUTES) < BILLING_TOL


def external_time_ok(row) -> bool:
    if "Has External Session" not in row.index:
        return True
    if not bool(row.get("Has External Session")):
        return False
    return (not pd.isna(row.get("_ExtStart_dt"))) and (not pd.isna(row.get("_ExtEnd_dt")))


def note_attendance_ok(row) -> bool:
    client_present = row.get("_Note_ClientPresent")
    bt_present = row.get("_Note_BTPresent")
    parent_present = row.get("_Note_ParentPresent")
    sibling_present = row.get("_Note_SiblingPresent")

    if all(pd.isna(x) for x in [client_present, bt_present, parent_present, sibling_present]):
        return True

    if not bool(client_present):
        return False
    if not bool(bt_present):
        return False
    if not (bool(parent_present) or bool(sibling_present)):
        return False
    return True


def note_parse_ok(row) -> bool:
    if "Note Parse PASS" not in row.index:
        return True
    val = row.get("Note Parse PASS")
    if pd.isna(val):
        return False
    return bool(val)


def sig_ok_base(row) -> bool:
    base_ts = row.get("_End_dt", pd.NaT)
    parent_sig_ts = row.get("_ParentSig_dt", pd.NaT)

    if pd.isna(parent_sig_ts):
        return False
    if pd.isna(base_ts):
        return False

    if USE_TIME_ADJ_OVERRIDE and has_time_adjust_sig(row):
        return True

    return within_time_tol(parent_sig_ts, base_ts, SIG_TOL_EARLY)


def daily_total_ok(row) -> bool:
    m = row.get("Daily Minutes")
    if pd.isna(m):
        return True
    return m < (DAILY_MAX_MINUTES + DAILY_TOL)


def evaluate_row(row) -> dict:
    dur_base = duration_ok_base(row)
    sig_base = sig_ok_base(row)
    ext_ok = external_time_ok(row)
    adj_sig = has_time_adjust_sig(row)
    daily_ok_val = daily_total_ok(row)
    note_ok_val = note_attendance_ok(row)
    note_parse_ok_val = note_parse_ok(row)

    duration_ok = dur_base
    sig_ok = sig_base

    overall = duration_ok and sig_ok and ext_ok and daily_ok_val and note_ok_val and note_parse_ok_val

    return {
        "duration_ok": duration_ok,
        "sig_ok": sig_ok,
        "ext_ok": ext_ok,
        "daily_ok": daily_ok_val,
        "note_ok": note_ok_val,
        "note_parse_ok": note_parse_ok_val,
        "has_time_adj_sig": adj_sig,
        "overall_pass": overall,
        "duration_ok_base": dur_base,
        "sig_ok_base": sig_base,
    }


def get_failure_reasons(row) -> str:
    eval_res = evaluate_row(row)
    reasons = []

    if not eval_res.get("note_parse_ok", True):
        note_pass = row.get("Note Parse PASS", np.nan)
        note_errs = row.get("Note Compliance Errors", np.nan)

        if pd.isna(note_pass):
            reasons.append("No matching session note found in PDF (required)")
        else:
            if pd.isna(note_errs) or str(note_errs).strip() == "":
                reasons.append("Session note failed compliance checks")
            else:
                reasons.append(f"Session note failed compliance checks: {note_errs}")

    if not eval_res["ext_ok"]:
        stime = row.get("Session Time", "")
        if pd.isna(stime) or str(stime).strip() == "":
            reasons.append("No Session time on note")
        else:
            reasons.append("Session Time invalid: must be same-day and end before 10:00 PM")

    if not eval_res["duration_ok"]:
        actual_min = row.get("Actual Minutes")
        if pd.isna(actual_min):
            reasons.append("Missing Duration data")
        else:
            reasons.append(
                f"Duration ({actual_min:.0f} min) must be between {MIN_MINUTES} and {MAX_MINUTES} minutes"
            )

    if not eval_res["sig_ok"]:
        if pd.isna(row.get("_ParentSig_dt", pd.NaT)):
            reasons.append("Missing Adult Caregiver signature time (required)")
        else:
            reasons.append("Parent signature too early.")

    if not eval_res.get("daily_ok", True):
        daily_min = row.get("Daily Minutes")
        if not pd.isna(daily_min):
            reasons.append(
                f"Total daily duration for this BT on {row.get('Date')} "
                f"({daily_min:.0f} min) exceeded"
            )

    if not eval_res.get("note_ok", True):
        missing = []

        client_ok = bool(row.get("_Note_ClientPresent"))
        bt_ok = bool(row.get("_Note_BTPresent"))
        parent_ok = bool(row.get("_Note_ParentPresent"))
        sibling_ok = bool(row.get("_Note_SiblingPresent"))

        if not client_ok:
            missing.append("Client")
        if not bt_ok:
            missing.append("BT/RBT")
        if not (parent_ok or sibling_ok):
            missing.append("Parent/Caregiver or Sibling")

        reasons.append("Session note attendance issue: missing " + ", ".join(missing))

    return "; ".join(reasons) if reasons else "PASS"


with tab2:
    st.header("Session Checker")

    st.subheader("Upload HiRamsus Excel File")
    sessions_file = st.file_uploader(
        "Upload HiRamsus Excel File",
        type=["xlsx", "xls"],
        key="sessions_file",
    )

    if not sessions_file:
        st.info("Upload the HiRamsus Excel file to continue.")
        st.stop()

    external_sessions_df = st.session_state.get("external_sessions_df", None)

    if external_sessions_df is not None and len(external_sessions_df) > 0:
        st.success(f"Using {len(external_sessions_df)} external session rows from the **Tools** tab.")
    else:
        st.warning(
            "No external sessions found from the Tools tab yet. "
            "You can upload an external session file manually."
        )
        manual_external_file = st.file_uploader(
            "External Session List (Client, Session Time)",
            type=["csv", "xlsx"],
            key="external_sessions_manual_only",
        )
        if manual_external_file is not None:
            external_sessions_df = read_any(manual_external_file)

    USE_TIME_ADJ_OVERRIDE = st.toggle(
        f"Use '{TIME_ADJ_COL}' as a signature override (only for sessions that failed signature timing; duration & daily limit still enforced)",
        value=True,
    )

    df = pd.read_excel(sessions_file, dtype=object)
    df = normalize_cols(df)

    if "Start date" not in df.columns and "Start Date" in df.columns:
        df = df.rename(columns={"Start Date": "Start date"})

    if not ensure_cols(df, REQ_COLS, "Sessions File"):
        st.stop()

    if USE_TIME_ADJ_OVERRIDE:
        if not ensure_cols(df, [TIME_ADJ_COL], "Sessions File"):
            st.stop()

    df_f = df[
        (df["Status"].astype(str).str.strip() == STATUS_REQUIRED)
        & (df["Session"].astype(str).str.strip() == SESSION_REQUIRED)
    ].copy()

    df_f["_RowOrder"] = np.arange(len(df_f))

    df_f["Actual Minutes"] = df_f["Duration"].apply(parse_duration_to_minutes)

    start_raw = df_f["Start date"].astype(str).str.strip()
    start_clean = start_raw.str.split().str[0]
    df_f["Date"] = pd.to_datetime(start_clean, errors="coerce").dt.strftime("%m/%d/%Y")

    df_f["_End_dt"] = pd.NaT
    df_f["_ParentSig_dt"] = pd.to_datetime(df_f["Adult Caregiver signature time"], errors="coerce")

    df_f["Staff Name"] = df_f["User"].apply(normalize_client_name_for_match)
    df_f["Client Name"] = df_f["Client"].apply(normalize_client_name_for_match)

    # ---- Matching unchanged ----
    if external_sessions_df is not None:
        ext_df = normalize_cols(external_sessions_df.copy())

        if ensure_cols(ext_df, ["Client", "Session Time"], "External Sessions File"):
            ext_df["Client Name"] = ext_df["Client"].apply(normalize_client_name_for_match)

            ext_df["SessionIndex"] = ext_df.groupby("Client Name").cumcount()
            df_f["SessionIndex"] = df_f.groupby("Client Name").cumcount()

            merge_cols = [
                "Client Name",
                "SessionIndex",
                "Session Time",
                "Session Date",
                "Present_Client",
                "Present_ParentCaregiver",
                "Present_Sibling",
                "Present_BT_RBT",
                "Note Parse PASS",
                "Note Compliance Errors",
            ]
            merge_cols = [c for c in merge_cols if c in ext_df.columns]

            df_f = df_f.merge(
                ext_df[merge_cols],
                on=["Client Name", "SessionIndex"],
                how="left",
                sort=False,
            )

            df_f = df_f.sort_values("_RowOrder", kind="mergesort").reset_index(drop=True)

            for src, dst in [
                ("Present_Client", "_Note_ClientPresent"),
                ("Present_ParentCaregiver", "_Note_ParentPresent"),
                ("Present_Sibling", "_Note_SiblingPresent"),
                ("Present_BT_RBT", "_Note_BTPresent"),
            ]:
                if src in df_f.columns:
                    df_f[dst] = df_f[src].fillna(False).astype(bool)
                else:
                    df_f[dst] = np.nan

            # ✅ Single authoritative normalization (no fallback keeping raw)
            if "Session Time" in df_f.columns:
                df_f["Session Time"] = df_f["Session Time"].apply(
            lambda v: v if isinstance(v, str) and ("AM" in v or "PM" in v)
            else normalize_session_time(v)
        )


            df_f["Has External Session"] = (
                df_f.get("Session Time", pd.Series([np.nan] * len(df_f))).notna()
                & (df_f.get("Session Time").astype(str).str.strip() != "")
            )

            df_f[["_ExtStart_dt", "_ExtEnd_dt"]] = df_f.apply(
                lambda r: pd.Series(parse_session_time_range(r.get("Session Time"), r.get("Date"))),
                axis=1,
            )

            has_ext_valid = (
                df_f["Has External Session"]
                & df_f["_ExtStart_dt"].notna()
                & df_f["_ExtEnd_dt"].notna()
            )

            df_f.loc[has_ext_valid, "Actual Minutes"] = (
                (df_f.loc[has_ext_valid, "_ExtEnd_dt"] - df_f.loc[has_ext_valid, "_ExtStart_dt"])
                .dt.total_seconds()
                / 60.0
            )
            df_f.loc[has_ext_valid, "_End_dt"] = df_f.loc[has_ext_valid, "_ExtEnd_dt"]

        else:
            st.warning("External sessions data provided, but required columns are missing.")

    # ---- BT contacts unchanged ----
    df_f["Phone"] = ""
    df_f["Email"] = ""

    if bt_contacts_file is not None:
        bt_df = read_any(bt_contacts_file)
        if bt_df is not None:
            bt_df = normalize_cols(bt_df)

            bt_required = {"BT Name", "Phone", "Email"}
            bt_missing = bt_required - set(bt_df.columns)
            if bt_missing:
                st.error(f"BT Contacts file is missing: {sorted(bt_missing)}")
            else:
                bt_df["BT_formatted"] = bt_df["BT Name"].apply(normalize_name)

                def norm_name(s: str) -> str:
                    s = str(s).strip().lower()
                    s = s.replace(",", " ")
                    s = " ".join(s.split())
                    return s

                bt_df["bt_norm"] = bt_df["BT_formatted"].apply(norm_name)

                staff_to_phone = {}
                staff_to_email = {}

                staff_unique = df_f["Staff Name"].dropna().unique()

                for staff in staff_unique:
                    staff_norm = norm_name(staff)
                    best_score = 0.0
                    best_row = None

                    for _, bt_row in bt_df.iterrows():
                        bt_name_norm = bt_row["bt_norm"]
                        score = SequenceMatcher(None, staff_norm, bt_name_norm).ratio()
                        if score > best_score:
                            best_score = score
                            best_row = bt_row

                    if best_row is not None and best_score >= 0.8:
                        staff_to_phone[staff] = best_row.get("Phone", "")
                        staff_to_email[staff] = best_row.get("Email", "")

                df_f["Phone"] = df_f["Staff Name"].map(staff_to_phone).fillna("")
                df_f["Email"] = df_f["Staff Name"].map(staff_to_email).fillna("")

    df_f["Daily Minutes"] = df_f.groupby(["Staff Name", "Date"])["Actual Minutes"].transform("sum")

    eval_results = df_f.apply(evaluate_row, axis=1, result_type="expand")
    df_f["_DurationOk"] = eval_results["duration_ok"]
    df_f["_SigOk"] = eval_results["sig_ok"]
    df_f["_ExtOk"] = eval_results["ext_ok"]
    df_f["_DailyOk"] = eval_results["daily_ok"]
    df_f["_HasTimeAdjSig"] = eval_results["has_time_adj_sig"]
    df_f["_NoteOk"] = eval_results["note_ok"]
    df_f["_NoteParseOk"] = eval_results["note_parse_ok"]
    df_f["_OverallPass"] = eval_results["overall_pass"]

    df_f["Failure Reasons"] = df_f.apply(get_failure_reasons, axis=1)

    for col in ["Adult Caregiver signature time"]:
        if col in df_f.columns:
            df_f[col] = (
                pd.to_datetime(df_f[col], errors="coerce")
                .dt.strftime("%m/%d/%Y %I:%M:%S %p")
                .fillna("")
            )

    display_cols = [
        "AlohaABA Appointment ID",
        "Date",
        "Client Name",
        "Staff Name",
        "Phone",
        "Email",
        "Duration",
        "Actual Minutes",
        "Daily Minutes",
        "Adult Caregiver signature time",
        "Session Time",
        "Note Compliance Errors",
    ]

    if TIME_ADJ_COL in df_f.columns:
        display_cols.append(TIME_ADJ_COL)

    display_cols.extend(
        [
            "_DurationOk",
            "_SigOk",
            "_ExtOk",
            "_DailyOk",
            "_HasTimeAdjSig",
            "_NoteOk",
            "_NoteParseOk",
            "Failure Reasons",
        ]
    )

    present_cols = [c for c in display_cols if c in df_f.columns]

    st.subheader("2) Results")
    with st.expander("Summary", expanded=False):
        st.dataframe(df_f[present_cols], use_container_width=True, height=560)

    st.caption("Summary")
    summary = pd.DataFrame(
        {
            "Total (after filters)": [len(df_f)],
            "Pass": [int(df_f["_OverallPass"].sum())],
            "Fail": [int((~df_f["_OverallPass"]).sum())],
        }
    )
    st.table(summary)

    if "Client Name" in df_f.columns:
        export_df = df_f[df_f["Client Name"] != "Marry Wang Demo"].copy()
    else:
        export_df = df_f.copy()

    st.session_state["session_checker_df"] = export_df.copy()
    st.session_state["session_checker_present_cols"] = present_cols

    xlsx_all = export_excel(export_df[present_cols])
    xlsx_clean = export_excel(export_df[export_df["_OverallPass"]][present_cols])
    xlsx_flagged = export_excel(export_df[~export_df["_OverallPass"]][present_cols])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ Download All", data=xlsx_all, file_name="all_sessions.xlsx")
    with c2:
        st.download_button("✅ Passed Only", data=xlsx_clean, file_name="clean_sessions.xlsx")
    with c3:
        st.download_button("⚠️ Failed Only", data=xlsx_flagged, file_name="flagged_sessions.xlsx")


# =========================================================
# TAB 3 (unchanged)
# =========================================================
with tab3:
    st.header("Billed Checker – Billed / Unbilled / No Match")

    base_df = st.session_state.get("session_checker_df")
    present_cols = st.session_state.get("session_checker_present_cols")

    if base_df is None or present_cols is None:
        st.info(
            "Please run the **Session Checker** in Tab 2 first. "
            "Once you have results there, come back here."
        )
    else:
        st.subheader("Upload Aloha Billing Status File")

        billing_file = st.file_uploader(
            "Upload Aloha file with billed & unbilled sessions "
            "(must include 'Appointment ID' and 'Date Billed')",
            type=["xlsx", "xls", "csv"],
            key="billing_status_file",
        )

        if billing_file is not None:
            try:
                billing_df = read_any(billing_file)
            except Exception as e:
                billing_df = None
                st.error(f"Error reading billing status file: {e}")

            if billing_df is not None:
                billing_df = normalize_cols(billing_df)

                if "Appointment ID" not in billing_df.columns or "Date Billed" not in billing_df.columns:
                    st.error("Billing status file must contain columns: 'Appointment ID' and 'Date Billed'.")
                elif "AlohaABA Appointment ID" not in base_df.columns:
                    st.error(
                        "Session Checker data is missing 'AlohaABA Appointment ID'. "
                        "Please ensure the HiRasmus export includes this column."
                    )
                else:
                    merged = base_df.merge(
                        billing_df[["Appointment ID", "Date Billed"]],
                        left_on="AlohaABA Appointment ID",
                        right_on="Appointment ID",
                        how="left",
                        sort=False,
                    )

                    def classify_status(row):
                        app_id = str(row.get("Appointment ID", "")).strip()
                        date_billed = str(row.get("Date Billed", "")).strip()

                        if app_id == "" or app_id.lower() == "nan":
                            return "No Match"

                        if date_billed != "" and date_billed.lower() != "nan":
                            return "Billed"
                        return "Unbilled"

                    merged["Billing Status"] = merged.apply(classify_status, axis=1)

                    all_df = merged.copy()
                    billed_df = merged[merged["Billing Status"] == "Billed"]
                    unbilled_df = merged[merged["Billing Status"] == "Unbilled"]
                    nomatch_df = merged[merged["Billing Status"] == "No Match"]

                    unbilled_clean_df = unbilled_df[unbilled_df["_OverallPass"] == True]
                    unbilled_flagged_df = unbilled_df[unbilled_df["_OverallPass"] == False]

                    total_sessions = len(merged)
                    billed_count = len(billed_df)
                    unbilled_clean_count = len(unbilled_clean_df)
                    unbilled_flagged_count = len(unbilled_flagged_df)
                    nomatch_count = len(nomatch_df)

                    st.markdown(
                        f"""
**Total sessions (from Session Checker):** {total_sessions}  
- **Billed:** {billed_count}  
- **Unbilled – Clean:** {unbilled_clean_count}  
- **Unbilled – Flagged:** {unbilled_flagged_count}  
- **No Match (no Appointment ID match):** {nomatch_count}
"""
                    )

                    summary_df = pd.DataFrame(
                        {
                            "Status": ["Billed", "Unbilled – Clean", "Unbilled – Flagged", "No Match"],
                            "Count": [billed_count, unbilled_clean_count, unbilled_flagged_count, nomatch_count],
                        }
                    )
                    st.table(summary_df)

                    dl_cols = [
                        c
                        for c in dict.fromkeys(
                            present_cols + ["Appointment ID", "Date Billed", "Billing Status"]
                        ).keys()
                        if c in merged.columns
                    ]

                    all_dl = all_df[dl_cols]
                    billed_dl = billed_df[dl_cols]
                    unbilled_clean_dl = unbilled_clean_df[dl_cols]
                    unbilled_flagged_dl = unbilled_flagged_df[dl_cols]
                    nomatch_dl = nomatch_df[dl_cols]

                    xlsx_all = export_excel(all_dl)
                    xlsx_billed = export_excel(billed_dl)
                    xlsx_unbilled_clean = export_excel(unbilled_clean_dl)
                    xlsx_unbilled_flagged = export_excel(unbilled_flagged_dl)
                    xlsx_nomatch = export_excel(nomatch_dl)

                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1:
                        st.download_button("⬇️ Billed Sessions", data=xlsx_billed, file_name="billed_sessions.xlsx")
                    with c2:
                        st.download_button("⬇️ Unbilled – Clean", data=xlsx_unbilled_clean, file_name="unbilled_clean_sessions.xlsx")
                    with c3:
                        st.download_button("⬇️ Unbilled – Flagged", data=xlsx_unbilled_flagged, file_name="unbilled_flagged_sessions.xlsx")
                    with c4:
                        st.download_button("⬇️ No Match Sessions", data=xlsx_nomatch, file_name="no_match_sessions.xlsx")
                    with c5:
                        st.download_button("⬇️ All Sessions", data=xlsx_all, file_name="all_sessions.xlsx")
